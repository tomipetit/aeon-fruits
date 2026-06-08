import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from game_state import GameState


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

_STATE_LABELS = {
    GameState.IDLE:         "READY  [SPACE to start]",
    GameState.ANIMAL:       "",
    GameState.FRUIT_SELECT: "Stand in your fruit area!",
    GameState.MIX:          "JUMP to mix!",
    GameState.RESULT:       "",
}


@dataclass
class _Particle:
    x: float
    y: float
    vy: float
    area_index: int


def _alpha_blend(base: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    cv2.addWeighted(overlay_bgr, alpha, base, 1.0 - alpha, 0, base)
    return base


def _draw_area_dividers(frame: np.ndarray):
    for i in range(1, config.NUM_AREAS):
        x = config.WIDTH * i // config.NUM_AREAS
        cv2.line(frame, (x, 0), (x, config.HEIGHT), (200, 200, 200), 3)


def _draw_animal_box(frame: np.ndarray, data: dict, small: bool = False):
    animal = data["animal"]
    if small:
        box_x, box_y, w, h = 40, 20, 560, 100
        name_scale, pref_scale = 1.6, 0.9
        name_y, pref_y = 70, 95
    else:
        box_x, box_y, w, h = 60, 120, 700, 160
        name_scale, pref_scale = 2.5, 1.2
        name_y, pref_y = box_y + 70, box_y + 140
    cv2.rectangle(frame, (box_x, box_y), (box_x + w, box_y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + w, box_y + h), (255, 255, 255), 3)
    cv2.putText(frame, animal["name"], (box_x + 20, name_y),
                _FONT_BOLD, name_scale, (255, 220, 100), 4, cv2.LINE_AA)
    cv2.putText(frame, animal["pref"], (box_x + 20, pref_y),
                _FONT, pref_scale, (220, 220, 220), 2, cv2.LINE_AA)


def _draw_area_counts(frame: np.ndarray, data: dict):
    area_w = config.WIDTH // config.NUM_AREAS
    for i, count in enumerate(data["area_counts"]):
        x = i * area_w + area_w // 2 - 30
        cv2.putText(frame, str(count), (x, 200), _FONT_BOLD, 3.0,
                    (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, str(count), (x, 200), _FONT_BOLD, 3.0,
                    (0, 0, 0), 2, cv2.LINE_AA)


def _draw_fruit_labels(frame: np.ndarray, data: dict):
    area_w = config.WIDTH // config.NUM_AREAS
    for i, fruit in enumerate(data["area_fruits"]):
        x = i * area_w + area_w // 2
        color = fruit["bgr"]
        cv2.circle(frame, (x, config.HEIGHT - 60), 40, color, -1)
        cv2.circle(frame, (x, config.HEIGHT - 60), 40, (255, 255, 255), 3)
        label = fruit["name"]
        tw, _ = cv2.getTextSize(label, _FONT, 1.0, 2)[0], None
        cv2.putText(frame, label, (x - tw[0] // 2, config.HEIGHT - 20),
                    _FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def _blended_juice_color(data: dict) -> tuple[int, int, int]:
    blended = np.zeros(3, dtype=np.float32)
    for i, prop in enumerate(data["fruit_proportions"]):
        blended += np.array(data["area_fruits"][i]["bgr"], dtype=np.float32) * prop
    return tuple(int(c) for c in blended)


def _draw_mixer(frame: np.ndarray, data: dict):
    juice_color = _blended_juice_color(data)
    mix_level = data["mix_level"]
    cx, cy, r = config.WIDTH // 2, config.HEIGHT // 2, 240

    # Outer ring
    cv2.circle(frame, (cx, cy), r, (80, 80, 80), -1)
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 6)

    # Juice fill (clip circle from bottom)
    fill_h = int(2 * r * mix_level)
    fill_y = cy + r - fill_h
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r - 6, 255, -1)
    juice = frame.copy()
    cv2.rectangle(juice, (cx - r, fill_y), (cx + r, cy + r), juice_color, -1)
    frame[mask > 0] = cv2.addWeighted(frame, 0.3, juice, 0.7, 0)[mask > 0]

    pct = int(mix_level * 100)
    cv2.putText(frame, f"{pct}%", (cx - 50, cy + 12),
                _FONT_BOLD, 2.0, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(frame, f"JUMP: {data['total_jumps']}", (cx - 80, cy + r + 60),
                _FONT, 1.4, (255, 255, 255), 3, cv2.LINE_AA)


def _draw_stars(frame: np.ndarray, data: dict):
    stars = data["star_rating"]
    star_size = 60
    total_w = stars * (star_size * 2 + 10)
    start_x = (config.WIDTH - total_w) // 2
    y = config.HEIGHT // 2 + 160

    for i in range(stars):
        cx = start_x + i * (star_size * 2 + 10) + star_size
        pts = []
        for j in range(5):
            outer = math.pi / 2 + j * 2 * math.pi / 5
            inner = outer + math.pi / 5
            pts.append([int(cx + star_size * math.cos(outer)),
                        int(y - star_size * math.sin(outer))])
            pts.append([int(cx + star_size * 0.4 * math.cos(inner)),
                        int(y - star_size * 0.4 * math.sin(inner))])
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts_arr], (0, 215, 255))
        cv2.polylines(frame, [pts_arr], True, (0, 165, 200), 2)

    score_pct = int(data["match_score"] * 100)
    cv2.putText(frame, f"MATCH {score_pct}%",
                (config.WIDTH // 2 - 130, config.HEIGHT // 2 + 280),
                _FONT_BOLD, 2.0, (255, 255, 255), 4, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, state: GameState):
    label = _STATE_LABELS.get(state, "")
    if not label:
        return
    cv2.putText(frame, label,
                (config.WIDTH // 2 - len(label) * 14, config.HEIGHT - 40),
                _FONT, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


class ARRenderer:
    def __init__(self):
        self._particles: list[_Particle] = []
        self._resting_counts: list[int] = [0] * config.NUM_AREAS
        self._spawn_accum: list[float] = [0.0] * config.NUM_AREAS
        self._prev_state: GameState | None = None

    def _reset(self):
        self._particles.clear()
        self._resting_counts = [0] * config.NUM_AREAS
        self._spawn_accum = [0.0] * config.NUM_AREAS

    def _spawn_particles(self, area_counts: list[int], area_fruits: list[dict]):
        area_w = config.WIDTH // config.NUM_AREAS
        total = sum(area_counts) or 1
        for i, cnt in enumerate(area_counts):
            rate = (cnt / total) * config.FRUIT_SPAWN_RATE / 30.0  # per frame at ~30fps
            self._spawn_accum[i] += rate
            while self._spawn_accum[i] >= 1.0:
                self._spawn_accum[i] -= 1.0
                x = random.uniform(i * area_w + 40, (i + 1) * area_w - 40)
                self._particles.append(_Particle(
                    x=x, y=-config.FRUIT_PARTICLE_RADIUS,
                    vy=config.FRUIT_PARTICLE_SPEED,
                    area_index=i,
                ))

    def _update_particles(self):
        alive = []
        rest_y = config.HEIGHT - 80
        for p in self._particles:
            p.y += p.vy
            if p.y >= rest_y:
                self._resting_counts[p.area_index] += 1
            else:
                alive.append(p)
        self._particles = alive

    def _draw_particles(self, frame: np.ndarray, area_fruits: list[dict]):
        r = config.FRUIT_PARTICLE_RADIUS
        for p in self._particles:
            color = area_fruits[p.area_index]["bgr"]
            cv2.circle(frame, (int(p.x), int(p.y)), r, color, -1)
            cv2.circle(frame, (int(p.x), int(p.y)), r, (255, 255, 255), 2)

    def _draw_resting_counts(self, frame: np.ndarray, data: dict):
        area_w = config.WIDTH // config.NUM_AREAS
        total = sum(self._resting_counts) or 1
        for i, cnt in enumerate(self._resting_counts):
            color = data["area_fruits"][i]["bgr"]
            x1, x2 = i * area_w, (i + 1) * area_w
            # Semi-transparent bar proportional to resting count
            bar_h = int((cnt / max(total, 1)) * 200)
            if bar_h > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1 + 5, config.HEIGHT - 80 - bar_h),
                               (x2 - 5, config.HEIGHT - 80), color, -1)
                _alpha_blend(frame, overlay, 0.45)
            cv2.putText(frame, str(cnt),
                        (x1 + area_w // 2 - 20, config.HEIGHT - 90),
                        _FONT_BOLD, 1.8, (255, 255, 255), 4, cv2.LINE_AA)

    def render(self, bgr_frame: np.ndarray, data: dict) -> np.ndarray:
        state: GameState = data["state"]

        # Reset particles at the start of each new round
        if self._prev_state != state and state == GameState.ANIMAL:
            self._reset()
        self._prev_state = state

        frame = bgr_frame.copy()
        _draw_area_dividers(frame)

        if state == GameState.IDLE:
            pass

        elif state == GameState.ANIMAL:
            _draw_animal_box(frame, data, small=False)

        elif state == GameState.FRUIT_SELECT:
            self._spawn_particles(data["area_counts"], data["area_fruits"])
            self._update_particles()
            self._draw_particles(frame, data["area_fruits"])
            self._draw_resting_counts(frame, data)
            _draw_fruit_labels(frame, data)
            _draw_area_counts(frame, data)
            _draw_animal_box(frame, data, small=True)

        elif state == GameState.MIX:
            _draw_mixer(frame, data)

        elif state == GameState.RESULT:
            # Show completed juice color as background tint
            juice_color = _blended_juice_color(data)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (config.WIDTH, config.HEIGHT), juice_color, -1)
            _alpha_blend(frame, overlay, 0.18)
            _draw_animal_box(frame, data, small=True)
            _draw_stars(frame, data)

        _draw_hud(frame, state)
        return frame
