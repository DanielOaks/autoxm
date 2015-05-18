#!/usr/bin/env python3
# AutoXM
# written by Daniel Oaks <daniel@danieloaks.net>
# released into the public domain
# inspired by the public domain autotracker by Ben "GreaseMonkey" Russell
#   and uses code from that script
import struct, random

# from musthe import scale, Note, Interval, Chord

# XM Module Handling
#

# constants
XM_TRACKER_NAME = 'AutoXM'
XM_ID_TEXT = 'Extended Module: '
XM_VERSION = 0x0104
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

    def save(self, filename):
        """Save the module as a file."""
        with open(filename, 'wb') as fp:
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
    def __init__(self, name='', relative_note=0):
        self.name = name

        self.bits = 8
        self.data = []

        self.loop_type = XM_LOOP_NONE
        self.loop_start = None
        self.loop_length = None

        self.volume = 0x40
        self.finetune = 0

        self.panning = 0  # -1 to 1
        self.relative_note = relative_note

        self.packing_type = XM_SAMPLE_PACKING_NONE

    def generate(self):
        raise Exception("The generate() function must be overridden in subclasses!")

    # def amplify(self):
    #     l = -0.0000000001
    #     h = 0.0000000001
    #     for v in self.data:#[len(self.data)//32:]:
    #         if v < l:
    #             l = v
    #         if v > h:
    #             h = v

    #     amp = self.boost / max(-l,h)
    #     #print amp

    #     for i in xrange(len(self.data)):
    #         self.data[i] *= amp


class XmInstrument:
    sample_generator = None

    def __init__(self, name='', *args, **kwargs):
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

        # should suffice for most instruments
        if self.sample_generator:
            self.samples.append(self.sample_generator(*args, **kwargs))

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


# actual instruments themselves
class NoiseSample(XmSample):
    def __init__(self, name='noise', relative_note=0, length=1, pattern='gauss'):
        """Noise sample.

        Arguments:
            length (float): Length of the sample in seconds
        """
        super().__init__(name='noise', relative_note=relative_note)

        self.sample_length = length
        self.sample_count = int(length * XM_SAMPLE_FREQ)

        self.loop_start = 0
        self.loop_length = self.sample_count
        self.loop_type = XM_LOOP_FORWARD

        self.pattern = pattern

    def generate(self):
        self.data = []

        for i in range(self.sample_count - 1):
            if self.pattern == 'gauss':
                val = random.gauss(0, 0.3)
                if val < -1:
                    val = -1
                elif val > 1:
                    val = 1
            else:
                val = random.random() * 2 - 1
            self.data.append(val)


class NoiseHit(XmInstrument):
    sample_generator = NoiseSample

    def __init__(self, name='noise', relative_note=0, length=1, fadeout=None):
        super().__init__(name=name, relative_note=relative_note, length=1)

        vol = 1

        if fadeout:
            self.volume_envelope.enable()
            self.volume_envelope.add_point(0, vol)
            self.volume_envelope.add_point(fadeout)


# Name Generation
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

    return mod


if __name__ == '__main__':
    chiptune = autoxm()

    bassdrum = NoiseHit('bassdrum', relative_note=XM_RELATIVE_OCTAVEDOWN - 6, fadeout=13)
    chiptune.add_instrument(bassdrum)

    snare = NoiseHit('snare', fadeout=9)
    chiptune.add_instrument(snare)

    pattern = XmPattern()
    chiptune.add_pattern_to_order(pattern)

    filename = chiptune.filename
    filename = 'new.xm'

    chiptune.save(filename)
    print('Saving as', filename)
