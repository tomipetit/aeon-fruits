import random
import time
from enum import Enum, auto

import config
import ai_comment


class GameState(Enum):
    IDLE = auto()
    ANIMAL = auto()
    FRUIT_SELECT = auto()
    MIX = auto()
    RESULT = auto()


class JuiceGame:
    def __init__(self):
        self.state = GameState.IDLE
        self._state_entered_at = time.monotonic()

        self._animal_index: int = 0
        self._session_animals: list[dict] = config.ANIMALS[:config.ANIMALS_PER_SESSION]
        self.animal: dict = self._session_animals[0]

        self.area_fruits: list[dict] = [
            config.FRUITS[self.animal["fruits"][i]] for i in range(config.NUM_AREAS)
        ]
        self.fruit_proportions: list[float] = [1.0 / config.NUM_AREAS] * config.NUM_AREAS
        self.mix_level: float = 0.0
        self._total_jumps: int = 0

        # Accumulated area counts during FRUIT_SELECT (used to compute proportions)
        self._area_accum: list[float] = [0.0] * config.NUM_AREAS
        self.match_score: float = 0.0
        self.star_rating: int = 0

        self._last_area_counts: list[int] = [0] * config.NUM_AREAS
        self.taste_comment: str = ""

    # ------------------------------------------------------------------ #

    def start(self):
        if self.state == GameState.IDLE:
            self._session_animals = random.sample(config.ANIMALS, config.ANIMALS_PER_SESSION)
            self._animal_index = 0
            self._enter(GameState.ANIMAL)

    def update(self, area_counts: list[int], jump_counts: list[int], spinning: bool):
        elapsed = time.monotonic() - self._state_entered_at
        self._last_area_counts = area_counts

        if self.state == GameState.IDLE:
            pass

        elif self.state == GameState.ANIMAL:
            if elapsed >= config.PHASE_DURATIONS["ANIMAL"]:
                self._enter(GameState.FRUIT_SELECT)

        elif self.state == GameState.FRUIT_SELECT:
            for i, cnt in enumerate(area_counts):
                self._area_accum[i] += cnt
            if elapsed >= config.PHASE_DURATIONS["FRUIT_SELECT"]:
                self._lock_fruit_proportions()
                self._enter(GameState.MIX)

        elif self.state == GameState.MIX:
            self._total_jumps += sum(jump_counts)
            self.mix_level = min(1.0, self._total_jumps / config.JUMPS_TO_MIX)
            done = self.mix_level >= 1.0 or elapsed >= config.PHASE_DURATIONS["MIX"]
            if done:
                self._calculate_score()
                self._enter(GameState.RESULT)

        elif self.state == GameState.RESULT:
            if elapsed >= config.PHASE_DURATIONS["RESULT"]:
                self._animal_index += 1
                if self._animal_index < len(self._session_animals):
                    self._enter(GameState.ANIMAL)
                else:
                    self._enter(GameState.IDLE)

    # ------------------------------------------------------------------ #

    def _enter(self, new_state: GameState):
        if new_state == GameState.ANIMAL:
            self.animal = self._session_animals[self._animal_index]
            self.area_fruits = [
                config.FRUITS[f] for f in self.animal["fruits"]
            ]
            self.fruit_proportions = [1.0 / config.NUM_AREAS] * config.NUM_AREAS
            self.mix_level = 0.0
            self._total_jumps = 0
            self._area_accum = [0.0] * config.NUM_AREAS
            self.taste_comment = ""
        elif new_state == GameState.MIX:
            # スコアを事前計算してから MIX 中に API 呼び出しを開始
            self._calculate_score()
            self.taste_comment = "ジュースの味は..."
            ai_comment.generate_taste_comment(
                self.animal,
                self.area_fruits,
                self.fruit_proportions,
                self.match_score,
                lambda text: setattr(self, "taste_comment", text),
            )
        elif new_state == GameState.RESULT:
            pass  # comment already generating since MIX started
        self.state = new_state
        self._state_entered_at = time.monotonic()

    def _lock_fruit_proportions(self):
        total = sum(self._area_accum)
        if total > 0:
            self.fruit_proportions = [v / total for v in self._area_accum]
        else:
            self.fruit_proportions = [1.0 / config.NUM_AREAS] * config.NUM_AREAS

    def _calculate_score(self):
        ideal = self.animal["ideal_mix"]
        diff = sum(abs(a - b) for a, b in zip(self.fruit_proportions, ideal))
        self.match_score = max(0.0, 1.0 - diff / 2.0)
        self.star_rating = max(1, round(self.match_score * 5))

    def get_render_data(self) -> dict:
        return {
            "state": self.state,
            "area_fruits": self.area_fruits[:],
            "fruit_proportions": self.fruit_proportions[:],
            "mix_level": self.mix_level,
            "total_jumps": self._total_jumps,
            "animal": self.animal,
            "match_score": self.match_score,
            "star_rating": self.star_rating,
            "area_counts": self._last_area_counts[:],
            "elapsed": time.monotonic() - self._state_entered_at,
            "taste_comment": self.taste_comment,
        }
