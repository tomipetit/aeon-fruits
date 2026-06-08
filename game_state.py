import random
import time
from enum import Enum, auto

import config


class GameState(Enum):
    IDLE = auto()
    ANIMAL = auto()
    AREA = auto()
    FILL = auto()
    MIX = auto()
    RESULT = auto()


class JuiceGame:
    def __init__(self):
        self.state = GameState.IDLE
        self._state_entered_at = time.monotonic()

        self.fill_levels = [0.0] * config.NUM_AREAS   # 0.0 – 1.0 per area
        self._fill_counters = [0] * config.NUM_AREAS   # raw jump counts
        self.area_fruits: list[dict] = [config.FRUITS[i % len(config.FRUITS)]
                                        for i in range(config.NUM_AREAS)]

        self.animal: dict = config.ANIMALS[0]
        self.match_score: float = 0.0           # 0.0 – 1.0
        self.star_rating: int = 0               # 1 – 5

        self._last_area_counts: list[int] = [0] * config.NUM_AREAS

    # ------------------------------------------------------------------ #

    def start(self):
        """Operator keypress to begin a round from IDLE."""
        if self.state == GameState.IDLE:
            self._enter(GameState.ANIMAL)

    def update(self, area_counts: list[int], jump_counts: list[int], spinning: bool):
        now = time.monotonic()
        elapsed = now - self._state_entered_at
        self._last_area_counts = area_counts

        if self.state == GameState.IDLE:
            pass  # wait for start()

        elif self.state == GameState.ANIMAL:
            if elapsed >= config.PHASE_DURATIONS["ANIMAL"]:
                self._enter(GameState.AREA)

        elif self.state == GameState.AREA:
            if elapsed >= config.PHASE_DURATIONS["AREA"]:
                self._assign_fruits(area_counts)
                self._enter(GameState.FILL)

        elif self.state == GameState.FILL:
            for i, jumps in enumerate(jump_counts):
                self._fill_counters[i] += jumps
                self.fill_levels[i] = min(1.0, self._fill_counters[i] / config.JUMPS_TO_FILL)
            all_full = all(lvl >= 1.0 for lvl in self.fill_levels)
            timeout = elapsed >= config.PHASE_DURATIONS["FILL"]
            if all_full or timeout:
                self._enter(GameState.MIX)

        elif self.state == GameState.MIX:
            if spinning or elapsed >= config.PHASE_DURATIONS["MIX"]:
                self._calculate_score()
                self._enter(GameState.RESULT)

        elif self.state == GameState.RESULT:
            if elapsed >= config.PHASE_DURATIONS["RESULT"]:
                self._enter(GameState.IDLE)

    # ------------------------------------------------------------------ #

    def _enter(self, new_state: GameState):
        if new_state == GameState.ANIMAL:
            self.animal = random.choice(config.ANIMALS)
            self.fill_levels = [0.0] * config.NUM_AREAS
            self._fill_counters = [0] * config.NUM_AREAS
        self.state = new_state
        self._state_entered_at = time.monotonic()

    def _assign_fruits(self, area_counts: list[int]):
        fruits = list(config.FRUITS)
        random.shuffle(fruits)
        for i in range(config.NUM_AREAS):
            self.area_fruits[i] = fruits[i % len(fruits)]

    def _calculate_score(self):
        total = sum(self.fill_levels) or 1.0
        actual = [lvl / total for lvl in self.fill_levels]
        ideal = self.animal["ideal_mix"]
        diff = sum(abs(a - b) for a, b in zip(actual, ideal))
        self.match_score = max(0.0, 1.0 - diff / 2.0)
        self.star_rating = max(1, round(self.match_score * 5))

    def get_render_data(self) -> dict:
        return {
            "state": self.state,
            "fill_levels": self.fill_levels[:],
            "area_fruits": self.area_fruits[:],
            "animal": self.animal,
            "match_score": self.match_score,
            "star_rating": self.star_rating,
            "area_counts": self._last_area_counts[:],
            "elapsed": time.monotonic() - self._state_entered_at,
        }
