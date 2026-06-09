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

# Fruit definitions (10 total): name → {name, bgr}
FRUITS: dict[str, dict] = {
    "いちご":             {"name": "いちご",             "bgr": (30,  30, 210)},
    "メロン":             {"name": "メロン",             "bgr": (60, 200,  80)},
    "かき":               {"name": "かき",               "bgr": (0,  100, 210)},
    "きょほう":           {"name": "きょほう",           "bgr": (90,   0, 110)},
    "シャインマスカット": {"name": "シャインマスカット", "bgr": (80, 210, 130)},
    "みかん":             {"name": "みかん",             "bgr": (0,  140, 255)},
    "ブルーベリー":       {"name": "ブルーベリー",       "bgr": (160,  50,  80)},
    "いちじく":           {"name": "いちじく",           "bgr": (60,   30, 130)},
    "もも":               {"name": "もも",               "bgr": (160, 180, 255)},
    "なし":               {"name": "なし",               "bgr": (100, 220, 190)},
}

# Animals with juice preferences (10 total; ANIMALS_PER_SESSION are chosen each session).
# fruits: exactly 3 fruit names (must exist in FRUITS) assigned to areas left→right.
# ideal_mix: ideal proportion [area0, area1, area2] summing to 1.0.
ANIMALS = [
    {
        "name": "うさぎ",
        "pref": "あまずっぱいのがすき！",
        "fruits": ["いちご", "みかん", "もも"],
        "ideal_mix": [0.5, 0.2, 0.3],
    },
    {
        "name": "くま",
        "pref": "あまいのがすき！",
        "fruits": ["もも", "メロン", "みかん"],
        "ideal_mix": [0.4, 0.4, 0.2],
    },
    {
        "name": "きつね",
        "pref": "つぶつぶがすき！",
        "fruits": ["きょほう", "シャインマスカット", "ブルーベリー"],
        "ideal_mix": [0.4, 0.3, 0.3],
    },
    {
        "name": "ぞう",
        "pref": "みずみずしいのがすき！",
        "fruits": ["メロン", "なし", "もも"],
        "ideal_mix": [0.4, 0.3, 0.3],
    },
    {
        "name": "パンダ",
        "pref": "さっぱりしたのがすき！",
        "fruits": ["メロン", "シャインマスカット", "なし"],
        "ideal_mix": [0.3, 0.4, 0.3],
    },
    {
        "name": "ライオン",
        "pref": "こくてあまいのがすき！",
        "fruits": ["かき", "みかん", "いちご"],
        "ideal_mix": [0.5, 0.3, 0.2],
    },
    {
        "name": "ねこ",
        "pref": "ちょっとすっぱいのがすき！",
        "fruits": ["いちご", "なし", "ブルーベリー"],
        "ideal_mix": [0.5, 0.2, 0.3],
    },
    {
        "name": "さる",
        "pref": "フルーティなのがすき！",
        "fruits": ["みかん", "いちご", "かき"],
        "ideal_mix": [0.4, 0.3, 0.3],
    },
    {
        "name": "ひつじ",
        "pref": "やさしいあまさがすき！",
        "fruits": ["もも", "いちご", "メロン"],
        "ideal_mix": [0.5, 0.3, 0.2],
    },
    {
        "name": "いぬ",
        "pref": "げんきになるのがすき！",
        "fruits": ["みかん", "きょほう", "いちじく"],
        "ideal_mix": [0.4, 0.3, 0.3],
    },
]

# Number of animals chosen per session
ANIMALS_PER_SESSION = 3
