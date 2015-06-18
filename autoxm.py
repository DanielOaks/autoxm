#!/usr/bin/env python3
# AutoXM
# written by Daniel Oaks <daniel@danieloaks.net>
# released into the public domain
# inspired by and uses code from the public domain
#   autotracker.py by Ben "GreaseMonkey" Russell
import math
import random
import struct

# from musthe import scale, Note, Interval, Chord

# XM Module Handling
#

# constants
XM_TRACKER_NAME = 'AutoXM'
XM_ID_TEXT = 'Extended Module: '
XM_VERSION = 0x0104
XM_ENVELOPE_FREQ = 50
XM_SAMPLE_FREQ = 8363  # assuming C-4, good enough

XM_RELATIVE_OCTAVEUP = 12
XM_RELATIVE_OCTAVEDOWN = -12

XM_SAMPLE_PACKING_NONE = 0x00
XM_SAMPLE_PACKING_ADPCM = 0xAD

XM_VIBRATO_NONE = 0x0

XM_BITFLAGS_16BIT = 0x8

XM_FLAG_AMIGA_FREQ = 0x0
XM_FLAG_LINEAR_FREQ = 0x1

XM_EFFECT_NONE = 0x0
XM_EFFECT_PARAMS_NONE = 0x0

XM_ENV_OFF = 0x0
XM_ENV_ON = 0x1
XM_ENV_SUSTAIN = 0x2
XM_ENV_LOOP = 0x2

XM_LOOP_NONE = 0x0
XM_LOOP_FORWARD = 0x1
XM_LOOP_PINGPONG = 0x2

MIDDLE_C = 220.0 * (2.0 ** (3.0 / 12.0))


# classes
class XmFile:
    """Stores and outputs an XM file."""
    def __init__(self, name, bpm, channels=2):
        self.name = name
        self.channels = []
        self.patterns = []
        self.pattern_order = []
        self.instruments = []
        self.flags = XM_FLAG_LINEAR_FREQ
        self.speed = 3
        self.bpm = bpm
        self.restart_position = 0

        self.tracker_name = XM_TRACKER_NAME
        self.version = XM_VERSION

        for i in range(channels):
            self.add_channel()

    @property
    def filename(self):
        return '{}.xm'.format(slugify(self.name))

    def add_instrument(self, instrument):
        new_instrument_id = len(self.instruments)
        self.instruments.append(instrument)

        return new_instrument_id

    def add_channel(self, volume=100, pan=0):
        new_channel = XmChannel(volume, pan)
        new_channel_id = len(self.channels)
        self.channels.append(new_channel)

        return new_channel_id

    def add_pattern(self, pattern):
        new_pattern_id = len(self.patterns)
        self.patterns.append(pattern)

        return new_pattern_id

    def add_pattern_to_order(self, pattern):
        if isinstance(pattern, XmPattern):
            if pattern not in self.patterns:
                self.add_pattern(pattern)

            pattern_id = self.patterns.index(pattern)
        else:
            pattern_id = pattern

        self.pattern_order.append(pattern_id)

    def _get_padded(self, in_b, length, pad_with=b'\0'):
        """Get the given bytes, truncated/padded to length."""
        # convert to bytes
        if isinstance(in_b, str):
            in_b = in_b.encode('ascii')
        elif isinstance(in_b, list):
            out_b = bytes()
            for element in in_b:
                if isinstance(element, str):
                    element = element.encode('ascii')
                out_b += bytes(element)

            in_b = out_b

        correct_bytes = in_b[:length]
        while len(correct_bytes) < length:
            correct_bytes += pad_with

        return correct_bytes

    def save(self):
        """Save the module as a file."""
        with open(self.filename, 'wb') as fp:
            # header
            fp.write(XM_ID_TEXT.encode('ascii'))
            fp.write(self._get_padded(self.name, 20, pad_with=b' '))
            fp.write(b'\x1a')
            fp.write(self._get_padded(XM_TRACKER_NAME, 20, pad_with=b' '))
            fp.write(struct.pack('<H', XM_VERSION))

            # we need to do this so we get correct xm header size
            xm_header = bytes()
            xm_header += struct.pack('<H', len(self.pattern_order))
            xm_header += struct.pack('<H', self.restart_position)
            xm_header += struct.pack('<H', len(self.channels) if len(self.channels) % 2 == 0 else len(self.channels) + 1)
            xm_header += struct.pack('<H', len(self.patterns))
            xm_header += struct.pack('<H', len(self.instruments))
            xm_header += struct.pack('<H', self.flags)
            xm_header += struct.pack('<H', self.speed)
            xm_header += struct.pack('<H', self.bpm)
            xm_header += self._get_padded(self.pattern_order, 256)

            # write header size and header itself
            header_size = len(xm_header) + 4
            fp.write(struct.pack('<I', header_size))
            fp.write(xm_header)

            # patterns
            for pattern in self.patterns:
                packed_pattern_data = bytes()
                for row in pattern.rows:
                    for chan in range(len(self.channels)):
                        note = row.notes.get(chan, None)
                        
                        if note is None:
                            # MSB indicates bit compression
                            packed_pattern_data += struct.pack('<B', 0x80)
                        else:
                            # XXX - to do MSB bit compression
                            packed_pattern_data += struct.pack('<B', note.xm_note)
                            packed_pattern_data += struct.pack('<B', note.instrument)
                            packed_pattern_data += struct.pack('<B', note.volume)
                            packed_pattern_data += struct.pack('<B', note.effect_type)
                            packed_pattern_data += struct.pack('<B', note.effect_params)

                packed_pattern_data_size = len(packed_pattern_data)

                # to get pattern header length
                pattern_header = bytes()
                pattern_header += struct.pack('<B', 0)  # packing type
                pattern_header += struct.pack('<H', len(pattern.rows))
                pattern_header += struct.pack('<H', packed_pattern_data_size)

                pattern_header_size = struct.pack('<I', len(pattern_header) + 4)

                fp.write(pattern_header_size)
                fp.write(pattern_header)
                fp.write(packed_pattern_data)

            # instruments
            for instrument in self.instruments:
                instrument_data = bytes()

                instrument_data += self._get_padded(instrument.name, 22)
                instrument_data += struct.pack('<B', 0)  # instrument type
                instrument_data += struct.pack('<H', len(instrument.samples))

                # sample headers
                sample_headers = bytes()

                for sample in instrument.samples:
                    sample.generate()

                    sample_headers += struct.pack('<I', len(sample.data))
                    sample_headers += struct.pack('<I', sample.loop_start)
                    sample_headers += struct.pack('<I', sample.loop_length)
                    sample_headers += struct.pack('<B', sample.volume)
                    sample_headers += struct.pack('<b', sample.finetune)

                    sample_type = sample.loop_type
                    if sample.bits == 16:
                        sample_type |= XM_BITFLAGS_16BIT

                    sample_headers += struct.pack('<B', sample_type)
                    sample_headers += struct.pack('<B', (sample.panning + 1) * 127)
                    sample_headers += struct.pack('<b', sample.relative_note)
                    sample_headers += struct.pack('<B', sample.packing_type)
                    sample_headers += self._get_padded(sample.name, 22)

                # second part of instrument headers
                if len(instrument.samples):
                    instrument_data += struct.pack('<I', len(sample_headers))

                    for sample_number in instrument.sample_map:
                        instrument_data += struct.pack('<B', sample_number)

                    for p in instrument.volume_envelope.padded_points:
                        instrument_data += struct.pack('<H', p.x)
                        instrument_data += struct.pack('<H', int(p.y * 0x40))
                    for p in instrument.panning_envelope.padded_points:
                        instrument_data += struct.pack('<H', p.x)
                        instrument_data += struct.pack('<H', int(((p.y + 1) / 2) * 0x40))

                    instrument_data += struct.pack('<B', len(instrument.volume_envelope.points))
                    instrument_data += struct.pack('<B', len(instrument.panning_envelope.points))
                    instrument_data += struct.pack('<B', instrument.volume_envelope.sustain_point)
                    instrument_data += struct.pack('<B', instrument.volume_envelope.loop_start_point)
                    instrument_data += struct.pack('<B', instrument.volume_envelope.loop_end_point)
                    instrument_data += struct.pack('<B', instrument.panning_envelope.sustain_point)
                    instrument_data += struct.pack('<B', instrument.panning_envelope.loop_start_point)
                    instrument_data += struct.pack('<B', instrument.panning_envelope.loop_end_point)
                    instrument_data += struct.pack('<B', instrument.volume_envelope.env_type)
                    instrument_data += struct.pack('<B', instrument.panning_envelope.env_type)
                    instrument_data += struct.pack('<B', instrument.vibrato_type)
                    instrument_data += struct.pack('<B', instrument.vibrato_sweep)
                    instrument_data += struct.pack('<B', instrument.vibrato_depth)
                    instrument_data += struct.pack('<B', instrument.vibrato_rate)
                    instrument_data += struct.pack('<H', instrument.volume_fadeout)
                    instrument_data += self._get_padded('', 22)  # reserved

                fp.write(struct.pack('<I', len(instrument_data) + 4))
                fp.write(instrument_data)

                # sample data
                fp.write(sample_headers)

                for sample in instrument.samples:
                    sample_data = bytes()

                    b1, b2, b3 = 0, 0, 0

                    # we store our sample data as -1 to 1 in-memory, so we convert it to
                    #   -127 to 127, doing the byte arithmetic wraparound ourselves
                    for point in sample.data:
                        b3 = int(point * 127)
                        b2 = b3 - b1
                        if b2 < -127:
                            tb3 = 128 + b3 + 127
                            b2 = tb3 - b1
                        if b2 > 127:
                            tb1 = 128 + b1 + 127
                            b2 = b3 - tb1
                        sample_data += struct.pack('<b', b2)
                        b1 = b3

                    fp.write(sample_data)


class XmEnvelopePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class XmEnvelope:
    def __init__(self):
        self.points = []

        self.sustain_point = 0
        self.loop_start_point = 0
        self.loop_end_point = 0
        self.env_type = XM_ENV_OFF

    def enable(self, sustain=False, loop=False):
        self.env_type = XM_ENV_ON
        if sustain:
            self.env_type |= XM_ENV_SUSTAIN
        if loop:
            self.env_type |= XM_ENV_LOOP

    def disable(self):
        self.env_type = XM_ENV_OFF

    def clear(self):
        self.points = []

    def add_point(self, x, y=0):
        self.points.append(XmEnvelopePoint(x, y))

    @property
    def padded_points(self):
        points = []

        for p in self.points:
            points.append(p)

        while len(points) < 12:
            points.append(XmEnvelopePoint(0, 0))

        return points


class XmSample:
    def __init__(self, name='', relative_note=0, length=0, filtl=0, filth=1):
        """Stores an XM instrument sample.

        Arguments:
            name (str): Name of the sample.
            relative_note (int): Note change relative to C-4 (-1 would be B-4).
            length (float): Length of the sample in seconds.
            filtl (int): Low-pass filter, 0 to 1.
            filth (int): High-pass filter, 0 to 1.
        """
        self.name = name

        self.sample_length = length
        self.sample_count = int(length * XM_SAMPLE_FREQ)

        self.bits = 8
        self.data = []

        self.loop_type = XM_LOOP_NONE
        self.loop_start = 0
        self.loop_length = 0

        self.volume = 0x40
        self.finetune = 0

        self.panning = 0  # -1 to 1
        self.relative_note = relative_note

        self.boost = 1.0

        self.filtl = filtl
        self.filth = filth

        self.packing_type = XM_SAMPLE_PACKING_NONE

    def clear(self):
        self.data = []
        self.ql = 0
        self.qh = 0

    def add_sample(self, val, filter=True):
        if filter:
            self.ql += (val - self.ql) * self.filtl
            self.qh += (val - self.qh) * self.filth
            val = self.ql - self.qh

        self.data.append(val)

    def generate(self):
        raise Exception("The generate() function must be overridden in subclasses!")

    def filt(self, in_data, filtl=0, filth=1):
        out_data = []

        ql = 0
        qh = 0

        for val in in_data:
            ql += (val - ql) * filtl
            qh += (val - qh) * filth
            val = ql - qh

            out_data.append(val)

        return out_data

    def amplify(self, in_data=None):
        if in_data:
            data = in_data
        else:
            data = self.data

        low = -0.0000000001
        high = 0.0000000001

        for val in data:
            if val < low:
                low = val
            if val > high:
                high = val

        amp = self.boost / max(-low, high)

        for i in range(len(data)):
            if in_data:
                data[i] *= amp
            else:
                self.data[i] *= amp

        return data


class XmInstrument:
    sample_generator = None

    def __init__(self, name='', length=1, fadeout=None, *args, **kwargs):
        self.name = name

        # samples
        self.samples = []
        self.sample_map = []

        # instrument info
        self.volume_envelope = XmEnvelope()
        self.panning_envelope = XmEnvelope()

        self.vibrato_type = XM_VIBRATO_NONE
        self.vibrato_sweep = 0
        self.vibrato_depth = 0
        self.vibrato_rate = 0

        self.volume_fadeout = 0

        # fadeout, envelope, defaults to linear
        vol = 1

        if fadeout:
            self.volume_envelope.enable()
            self.volume_envelope.add_point(0, vol)
            self.volume_envelope.add_point(int(fadeout * XM_ENVELOPE_FREQ))

        # should suffice for most instruments
        if self.sample_generator:
            self.samples.append(self.sample_generator(length=length, *args, **kwargs))

    @property
    def sample_map(self):
        return self._sample_map

    @sample_map.setter
    def sample_map(self, value):
        while len(value) < 96:
            value.append(0)

        self._sample_map = value


class XmChannel:
    """Stores an XM channel."""
    def __init__(self, volume=100, pan=0):
        self.volume = volume
        self.pan = pan


class XmPattern:
    """Stores an XM pattern."""
    def __init__(self, number_of_rows=0x80):
        self.rows = []

        while len(self.rows) < number_of_rows:
            empty_row = XmRow()
            self.add_row(empty_row)

    @property
    def number_of_rows(self):
        return len(rows)

    def add_row(self, row):
        self.rows.append(row)


class XmRow:
    """Stores an XM row."""
    def __init__(self):
        self.notes = {}

    def set_note(self, channel, note, instrument, volume, effect_type=XM_EFFECT_NONE, effect_params=XM_EFFECT_PARAMS_NONE):
        new_note = XmNote(note, instrument, volume, effect_type, effect_params)
        self.notes[channel] = new_note


class XmNote:
    """Stores an XM note."""
    def __init__(self, note, instrument, volume, effect_type, effect_params):
        self.note = note
        self.instrument = instrument
        self.volume = volume
        self.effect_type = effect_type
        self.effect_params = effect_params

    @property
    def xm_note(self):
        return note


## Noise Functions
perm = [
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,
    142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,
    203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,
    220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,
    132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
    186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,
    59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,
    70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,
    178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
    241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
    176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,
    128,195,78,66,215,61,156,180,151,160,137,91,90,15,131,13,201,95,96,53,194,
    233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,
    75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,
    20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,
    111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,
    63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,188,
    159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,147,
    118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,
    213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,
    19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,242,193,
    238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,214,31,
    181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,
    222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
]


def grad(hash, x):
    h = int(hash) & 15
    grad = 1 + (h & 7)  # gradient value 1, 2, ..., 8
    if h & 8:
        grad = - grad  # set a random sign for the gradient
    return grad * x  # multiply the gradient with the distance


def simplex_noise_1d(x):
    i0 = math.floor(x)
    i1 = i0 + 1
    x0 = x - i0
    x1 = x0 - 1

    t0 = 1 - x0 * x0
    t0 *= t0
    n0 = t0 * t0 * grad(perm[i0 & 0xff], x0)

    t1 = 1 - x1 * x1
    t1 *= t1
    n1 = t1 * t1 * grad(perm[i1 & 0xff], x1)

    # the maximum value of this noise is 8*(3/4)^4 = 2.53125
    # a factor of 0.395 scales to fit exactly within [-1,1]
    return 0.395 * (n0 + n1)


## Instruments
#

# noise
class NoiseSample(XmSample):
    def __init__(self, name='noise', pattern='gauss', **kwargs):
        """Noise sample.

        Arguments:
            length (float): Length of the sample in seconds
        """
        super().__init__(name=name, **kwargs)

        self.loop_start = 0
        self.loop_length = self.sample_count
        self.loop_type = XM_LOOP_FORWARD

        self.pattern = pattern

    def generate(self):
        self.clear()

        for i in range(self.sample_count):
            if self.pattern == 'gauss':
                val = random.gauss(0, 0.3)
                if val < -1:
                    val = -1
                elif val > 1:
                    val = 1
            else:
                val = random.random() * 2 - 1

            self.add_sample(val)

        self.amplify()


class NoiseHit(XmInstrument):
    sample_generator = NoiseSample

    def __init__(self, name='noise', **kwargs):
        super().__init__(name=name, **kwargs)

# kick
class KickSample(XmSample):
    def __init__(self, name='kick', **kwargs):
        """Kick sample.

        Arguments:
            length (float): Length of the sample in seconds
        """
        super().__init__(name=name, **kwargs)

    def generate(self):
        self.clear()

        vol_noise = 9
        vol_sine = 15
        vol_noise_decay = 1.0 / (XM_SAMPLE_FREQ * 0.004)
        vol_sine_decay = 1.0 / (XM_SAMPLE_FREQ * 0.022)

        max_sin = 0.6

        q_noise = 0  # XXX - quantise noise?

        kick_mul = math.pi * 2 * 153 / XM_SAMPLE_FREQ
        offs_sine = 0
        offs_sine_speed = kick_mul / 2.5
        offs_sine_decay = 0.9992

        simplex_seed = random.randint(1, 10000)

        for i in range(self.sample_count):
            sv = max(-(max_sin), min(max_sin, math.sin(offs_sine)))
            offs_sine += offs_sine_speed
            offs_sine_speed *= offs_sine_decay

            nv = random.random() * 2 - 1
            q_noise += (nv - q_noise) * 0.1
            nv = q_noise

            noise = nv * vol_noise
            sine = sv * vol_sine

            smp = noise + sine / 2

            # we go to simplex noise after 6pi, because it sounds better
            #   and stops it from going into a highish-pitched 'buzz'
            if offs_sine > 12 * math.pi:
                noise_smp = simplex_noise_1d(offs_sine * (i / self.sample_count) * 1.35 + simplex_seed) * 0.7

            if offs_sine > 12 * math.pi and offs_sine < 13 * math.pi:
                # mix sine and simplex together for a bit
                mix = (offs_sine - 12 * math.pi) / math.pi
                smp = ((1 - mix) * smp) + (mix * noise_smp)
            elif offs_sine >= 13 * math.pi:
                smp = noise_smp

            # vol makes sure it fades off nicely
            vol = min(1, (self.sample_count - max(1, i)) / self.sample_count * 2)
            self.add_sample(smp * vol)

            vol_noise -= vol_noise_decay
            if vol_noise < 0:
                vol_noise = 0

            vol_sine -= vol_sine_decay
            if vol_sine < 0:
                vol_sine = 0

        self.data = self.filt(self.data, filth=0.85)

        self.amplify()


class KickHit(XmInstrument):
    sample_generator = KickSample

    def __init__(self, name='kick', length=0.24, **kwargs):
        super().__init__(name=name, length=length, **kwargs)


# Strings
class KsSample(XmSample):
    def __init__(self, name='ks synth', noise_filtl=0, noise_filth=1, **kwargs):
        """Karplus–Strong string synthesis."""
        super().__init__(name=name, **kwargs)

        self.noise_filtl = noise_filtl
        self.noise_filth = noise_filth

        self.loop_type = XM_LOOP_PINGPONG

    def generate(self):
        self.clear()

        # variables
        freq = MIDDLE_C
        decay = 0.02

        # we use 10000 instead of 1000 because our math works out better like
        #   that. haven't had a good look into it, but it sounds about right
        period = int(10000 / freq)
        # noise defaults to 5ms, with a minimum of the period
        noise_len = int((5 / 1000) * XM_SAMPLE_FREQ)
        if noise_len < period:
            noise_len = period

        loss_factor = 1 - decay
        stretch = 0.3

        # generate and center our random samples
        noise = []
        for i in range(noise_len):
            # gaussian sounds better than completely random for this
            val = random.gauss(0, 0.5)

            if val < -1:
                val = -1
            elif val > 1:
                val = 1

            noise.append(val)
        # filter, particularly the lowpass filter, cleans up lots of the ugly
        #   plucking noise we don't really want to leave in
        noise = self.filt(noise, filtl=self.noise_filtl, filth=self.noise_filth)
        noise = self.amplify(noise)

        avg = 0
        count = 0
        for j in range(noise_len):
            avg += noise[j]
            count += 1
        avg /= count

        # correct for the bias
        for j in range(noise_len):
            noise[j] = noise[j] - avg

            if 1 < noise[j]:
                noise[j] = 1
            elif -1 > noise[j]:
                noise[j] = -1

        # calculate samples
        for i in range(self.sample_count):
            if i < noise_len:
                pos = i
                data1 = noise[pos]
                data2 = noise[pos - 1]
            else:
                pos = i - period
                data1 = self.data[pos]
                data2 = self.data[pos - 1]

            val = ((1 - stretch) * data1) + (stretch * data2)
            val *= loss_factor

            self.add_sample(val)

        # set loop
        self.loop_start = self.sample_count - period
        self.loop_length = period

        self.amplify()


class KsInstrument(XmInstrument):
    sample_generator = KsSample

    def __init__(self, name='ks', length=2, **kwargs):
        super().__init__(name=name, length=length, **kwargs)


# Name Generation
#

# Taken from autotracker.py and extended
def slugify(name):
    """Take a generated name, output an autoxm slug."""
    slug = name.lower().replace(' ', '-').replace('\t', '-').replace('--', '').replace("'", '')
    return slug

NAME_NOUNS = [
    ('cat', 'cats'), ('kitten', 'kittens'),
    ('dog', 'dogs'), ('puppy', 'puppies'),
    ('elf', 'elves'), ('knight', 'knights'),
    ('wizard', 'wizards'), ('witch', 'witches'), ('leprechaun', 'leprechauns'),
    ('dwarf', 'dwarves'), ('golem', 'golems'), ('troll', 'trolls'),
    ('city', 'cities'), ('castle', 'castles'), ('town', 'towns'), ('village', 'villages'),
    ('journey', 'journeys'), ('flight', 'flights'), ('place', 'places'),
    ('bird', 'birds'),
    ('ocean', 'oceans'), ('sea', 'seas'),
    ('boat', 'boats'), ('ship', 'ships'),
    ('whale', 'whales'),
    ('brother', 'brothers'), ('sister', 'sisters'),
    ('viking', 'vikings'), ('ghost', 'ghosts'),
    ('garden', 'gardens'), ('park', 'parks'),
    ('forest', 'forests'), ('ogre', 'ogres'),
    ('sweet', 'sweets'), ('candy', 'candies'),
    ('hand', 'hands'), ('foot', 'feet'), ('arm', 'arms'), ('leg', 'legs'),
    ('body', 'bodies'), ('head', 'heads'), ('wing', 'wings'),
    ('gorilla', 'gorillas'), ('ninja', 'ninjas'), ('bear', 'bears'),
    ('vertex', 'vertices'), ('matrix', 'matrices'), ('simplex', 'simplices'),
    ('shape', 'shapes'),
    ('apple', 'apples'), ('pear', 'pears'), ('banana', 'bananas'),
    ('orange', 'oranges'),
    ('demoscene', 'demoscenes'),
    ('sword', 'swords'), ('shield', 'shields'), ('gun', 'guns'), ('cannon', 'cannons'),
    ('report', 'reports'), ('sign', 'signs'), ('age', 'ages'),
    ('blood', 'bloods'), ('breed', 'breeds'), ('monument', 'monuments'),
    ('cheese', 'cheeses'), ('horse', 'horses'), ('sheep', 'sheep'), ('fish', 'fish'),
    ('dock', 'docks'), ('tube', 'tubes'), ('road', 'roads'), ('path', 'paths'),
    ('tunnel', 'tunnels'), ('resort', 'resorts'),
    ('toaster', 'toasters'), ('goat', 'goats'),
    ('tofu', 'tofus'), ('vine', 'vines'), ('branch', 'branches'),
    ('atom', 'atoms'), ('train', 'trains'), ('plane', 'planes'),
]

NAME_VERBS = [
    'building', 'flying', 'baking', 'writing', 'tracking', 'exploring',
    'walking', 'running', 'flying', 'eating', 'licking', 'designing',
    'deceiving', 'greeting', 'graduating', 'enduring', 'enforcing',
]

NAME_ADVERBS = [
    'pleasingly', 'absurdly', 'offensively', 'crazily', 'magically',
    'deliciously', 'randomly', 'woefully', 'tearfully', 'poorly',
    'cowardly', 'concerningly',
]

NAME_ADJECTIVES = [
    'tense', 'grand', 'pleasing', 'absurd', 'offensive', 'crazed',
    'magic', 'lovely', 'tired', 'lively', 'tasty', 'jealous',
    'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown',
    'white', 'black', 'cheap', 'blazed', 'biased', 'sweet',
    'invisible', 'hidden', 'secret', 'long', 'short', 'tall', 'broken',
    'random', 'fighting', 'hunting', 'eating', 'drinking', 'drunk',
    'weary', 'strong', 'weak', 'woeful', 'tearful', 'rich', 'poor',
    'awoken', 'sacred', 'clumsy', 'mysterious', 'obnoxious', 'panicky',
    'magnificent', 'quaint', 'important', 'powerful', 'shy', 'wrong',
    'melodic', 'noisy', 'thundering', 'deafening', 'gentle', 'delightful',
    'eager', 'faithful', 'tasteless', 'modern', 'quick', 'slow', 'bruised',
    'contained', 'engineered',
]

NAME_PATTERNS = [
    "(a) (ns) of (ns)",
    "(a) (ns?)",
    "(a) and (a)",
    "(av) (a) (ns?)",
    "(av) (a) (ns?)",
    "(N)'s (ns?)",
    "(Ns?) and (Ns?)",
    "(ns?) of (Ns?)",
    "(ns?) of the (ns?)",
    "(v) (ns)",
    "(v) all of the (ns)",
    "on the (n)'s (ns?)",
    "the (a) (ns?)",
    "the (av) (a) (ns?)",
    "the (n)'s (ns?)",
    "the (v) (ns?)",
]

def autoname():
    """Generate a module name."""
    pattern = random.choice(NAME_PATTERNS)
    name = ''

    while len(pattern):
        # if next 'character' is a variable, insert that
        var = True
        if pattern.startswith('(A)'):
            name += random.choice(NAME_ADJECTIVES).capitalize()
        elif pattern.startswith('(a)'):
            name += random.choice(NAME_ADJECTIVES)
        elif pattern.startswith('(V)'):
            name += random.choice(NAME_VERBS).capitalize()
        elif pattern.startswith('(v)'):
            name += random.choice(NAME_VERBS)
        elif pattern.startswith('(AV)'):
            name += random.choice(NAME_ADVERBS).capitalize()
        elif pattern.startswith('(av)'):
            name += random.choice(NAME_ADVERBS)
        elif pattern.startswith('(N)'):
            name += random.choice(NAME_NOUNS)[0].capitalize()
        elif pattern.startswith('(n)'):
            name += random.choice(NAME_NOUNS)[0]
        elif pattern.startswith('(Ns)'):
            name += random.choice(NAME_NOUNS)[1].capitalize()
        elif pattern.startswith('(ns)'):
            name += random.choice(NAME_NOUNS)[1]
        elif pattern.startswith('(Ns?)'):
            name += random.choice(random.choice(NAME_NOUNS)).capitalize()
        elif pattern.startswith('(ns?)'):
            name += random.choice(random.choice(NAME_NOUNS))
        else:
            # if not a variable, just insert character directly
            name += pattern[0]
            pattern = pattern[1:]
            var = False

        # if it was a variable, cull that whole 'character' from the pattern
        if var:
            pattern = pattern.split(')', 1)[1]

    return name


# Music Generation
#
def autoxm(name=None, tempo=None):
    """Automatically generate an XmFile."""
    if name is None:
        name = autoname()
    if tempo is None:
        tempo = random.randint(90, 160)
    mod = XmFile(name, tempo)

    # adding instruments
    kick = KickHit('kick', filth=0.79)
    mod.add_instrument(kick)

    string = KsInstrument('string', length=2, fadeout=12, filtl=0.003, filth=0.92, noise_filth=0.02)
    mod.add_instrument(string)

    hatclosed = NoiseHit('highhat closed', relative_note=XM_RELATIVE_OCTAVEUP + 6, fadeout=0.1, filtl=0.99, filth=0.20)
    mod.add_instrument(hatclosed)

    hatopen = NoiseHit('highhat open', relative_note=XM_RELATIVE_OCTAVEUP + 6, fadeout=0.225, filtl=0.99, filth=0.20)
    mod.add_instrument(hatopen)

    snare = NoiseHit('snare', relative_note=3, fadeout=0.122, filtl=0.27, filth=0.44)
    mod.add_instrument(snare)

    # add basic pattern so we can open the file
    pattern = XmPattern()
    mod.add_pattern_to_order(pattern)

    return mod


if __name__ == '__main__':
    chiptune = autoxm('new')

    # saving out
    chiptune.save()

    print('Saving as', chiptune.filename)
