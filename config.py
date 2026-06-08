WIDTH = 1920
HEIGHT = 1080

NUM_AREAS = 3

# NDI source: set to None to show interactive prompt, or set index (0-based) to auto-select
NDI_SOURCE_INDEX = None

# HDMI second monitor: set X offset if projector is a second monitor (e.g. 1920 for right-of-primary)
DISPLAY_MONITOR_OFFSET_X = 0

# Detection processing scale (0.5 = half resolution for MOG2, faster but less precise)
DETECTION_SCALE = 0.5

# Background subtractor
MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 25

# Minimum foreground blob area in pixels (at detection scale)
MIN_PERSON_BLOB_AREA = 2000

# Jump detection
JUMP_Y_THRESHOLD = 0.04   # normalized Y displacement (fraction of frame height)
JUMP_DEBOUNCE_SEC = 0.5   # seconds between jump counts per blob

# Spin detection (optical flow)
SPIN_FLOW_THRESHOLD = 3.0   # mean absolute horizontal flow to trigger spin
SPIN_DURATION_SEC = 2.0     # how long flow must exceed threshold

# Game phase durations (seconds)
PHASE_DURATIONS = {
    "IDLE": -1,             # wait for operator keypress
    "ANIMAL": 4,
    "FRUIT_SELECT": 15,     # stand in areas, fruits fall proportionally
    "MIX": 15,              # everyone jumps to mix
    "RESULT": 6,
}

# Total jumps (all areas combined) to reach mix_level=1.0
JUMPS_TO_MIX = 50

# Fruit particle visual settings
FRUIT_SPAWN_RATE = 2.0      # fruits per second for a fully-occupied area
FRUIT_PARTICLE_SPEED = 12   # fall speed in pixels per frame (at full resolution)
FRUIT_PARTICLE_RADIUS = 28  # particle circle radius in pixels

# AR overlay alpha (0.0 = invisible, 1.0 = opaque)
FILL_OVERLAY_ALPHA = 0.5

# Area boundary dead zone (fraction of frame width on each side of boundary)
AREA_DEAD_ZONE = 0.03

# Fruit definitions: name, BGR color for overlay, ideal mix ratios per animal
FRUITS = [
    {"name": "オレンジ", "bgr": (0, 165, 255)},
    {"name": "ぶどう",   "bgr": (128, 0, 128)},
    {"name": "メロン",   "bgr": (0, 200, 100)},
]

# Animals with juice preferences
ANIMALS = [
    {
        "name": "うさぎ",
        "pref": "あまずっぱいのがすき！",
        "ideal_mix": [0.5, 0.2, 0.3],
    },
    {
        "name": "くま",
        "pref": "あまいのがすき！",
        "ideal_mix": [0.6, 0.3, 0.1],
    },
    {
        "name": "きつね",
        "pref": "つぶつぶがすき！",
        "ideal_mix": [0.3, 0.4, 0.3],
    },
]
