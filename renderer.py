import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from game_state import GameState


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

_FONT_JA_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
_font_ja_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font_ja(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_ja_cache:
        _font_ja_cache[size] = ImageFont.truetype(_FONT_JA_PATH, size)
    return _font_ja_cache[size]


def _draw_ja_texts(frame: np.ndarray,
                   texts: list[tuple[str, int, int, int, tuple, str]]) -> None:
    """Batch-draw Japanese texts with a single BGR↔RGB conversion."""
    if not texts:
        return
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, x, y, size, bgr, anchor in texts:
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        draw.text((x, y), text, font=_get_font_ja(size), fill=rgb, anchor=anchor)
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

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


def _draw_animal_box(frame: np.ndarray, data: dict, small: bool = False) -> list:
    animal = data["animal"]
    if small:
        box_x, box_y, w, h = 40, 20, 560, 100
        name_size, pref_size = 42, 24
        name_y, pref_y = 68, 92
    else:
        box_x, box_y, w, h = 60, 120, 700, 160
        name_size, pref_size = 62, 32
        name_y, pref_y = box_y + 68, box_y + 138
    cv2.rectangle(frame, (box_x, box_y), (box_x + w, box_y + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + w, box_y + h), (255, 255, 255), 3)
    return [
        (animal["name"], box_x + 20, name_y, name_size, (255, 220, 100), "ls"),
        (animal["pref"], box_x + 20, pref_y, pref_size, (220, 220, 220), "ls"),
    ]


def _draw_area_counts(frame: np.ndarray, data: dict):
    area_w = config.WIDTH // config.NUM_AREAS
    for i, count in enumerate(data["area_counts"]):
        x = i * area_w + area_w // 2 - 30
        cv2.putText(frame, str(count), (x, 200), _FONT_BOLD, 3.0,
                    (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, str(count), (x, 200), _FONT_BOLD, 3.0,
                    (0, 0, 0), 2, cv2.LINE_AA)


def _draw_fruit_labels(frame: np.ndarray, data: dict) -> list:
    area_w = config.WIDTH // config.NUM_AREAS
    ja_texts = []
    for i, fruit in enumerate(data["area_fruits"]):
        x = i * area_w + area_w // 2
        color = fruit["bgr"]
        cv2.circle(frame, (x, config.HEIGHT - 60), 40, color, -1)
        cv2.circle(frame, (x, config.HEIGHT - 60), 40, (255, 255, 255), 3)
        ja_texts.append((fruit["name"], x, config.HEIGHT - 18, 30, (255, 255, 255), "ms"))
    return ja_texts


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


def _wrap_ja(text: str, max_chars: int = 22) -> list[str]:
    """Split Japanese text into lines of at most max_chars characters."""
    lines: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if len(current) >= max_chars and ch in "。、！？♪\n":
            lines.append(current.strip())
            current = ""
    if current.strip():
        lines.append(current.strip())
    return lines or [text]


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


_ICON_RADIUS_MIN = 60.0
_ICON_RADIUS_MAX = 260.0
_ICON_LERP = 0.10  # smoothing factor per frame


class ARRenderer:
    def __init__(self):
        self._particles: list[_Particle] = []
        self._resting_counts: list[int] = [0] * config.NUM_AREAS
        self._spawn_accum: list[float] = [0.0] * config.NUM_AREAS
        self._smooth_radii: list[float] = [0.0] * config.NUM_AREAS
        self._prev_state: GameState | None = None

    def _reset(self):
        self._particles.clear()
        self._resting_counts = [0] * config.NUM_AREAS
        self._spawn_accum = [0.0] * config.NUM_AREAS
        self._smooth_radii = [0.0] * config.NUM_AREAS

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

        ja_texts: list = []

        if state == GameState.IDLE:
            pass

        elif state == GameState.ANIMAL:
            ja_texts += _draw_animal_box(frame, data, small=False)

        elif state == GameState.FRUIT_SELECT:
            self._spawn_particles(data["area_counts"], data["area_fruits"])
            self._update_particles()

            # Update smooth radii toward target based on accumulated count
            for i, cnt in enumerate(self._resting_counts):
                target = _ICON_RADIUS_MIN + (_ICON_RADIUS_MAX - _ICON_RADIUS_MIN) * min(cnt / 50.0, 1.0)
                self._smooth_radii[i] += (target - self._smooth_radii[i]) * _ICON_LERP

            # Draw centered fruit icons with count inside
            area_w = config.WIDTH // config.NUM_AREAS
            for i, fruit in enumerate(data["area_fruits"]):
                cx = i * area_w + area_w // 2
                cy = config.HEIGHT // 2
                r = max(2, int(self._smooth_radii[i]))
                color = fruit["bgr"]
                cv2.circle(frame, (cx, cy), r, color, -1)
                cv2.circle(frame, (cx, cy), r, (255, 255, 255), max(3, r // 25))
                # Count number centered inside the icon
                if r > 40:
                    count_str = str(self._resting_counts[i])
                    fscale = max(1.2, r / 90.0) * 2.0
                    thickness = max(4, r // 30)
                    (tw, th), _ = cv2.getTextSize(count_str, _FONT_BOLD, fscale, thickness)
                    tx, ty = cx - tw // 2, cy + th // 2
                    cv2.putText(frame, count_str, (tx, ty), _FONT_BOLD, fscale,
                                (255, 255, 255), thickness + 4, cv2.LINE_AA)
                    cv2.putText(frame, count_str, (tx, ty), _FONT_BOLD, fscale,
                                (30, 30, 30), thickness, cv2.LINE_AA)

            self._draw_particles(frame, data["area_fruits"])
            ja_texts += _draw_fruit_labels(frame, data)
            ja_texts += _draw_animal_box(frame, data, small=True)

        elif state == GameState.MIX:
            _draw_mixer(frame, data)

        elif state == GameState.RESULT:
            # Show completed juice color as background tint
            juice_color = _blended_juice_color(data)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (config.WIDTH, config.HEIGHT), juice_color, -1)
            _alpha_blend(frame, overlay, 0.18)
            ja_texts += _draw_animal_box(frame, data, small=True)
            _draw_stars(frame, data)
            # AI taste comment
            comment = data.get("taste_comment", "")
            if comment:
                cx = config.WIDTH // 2
                base_y = config.HEIGHT // 2 + 330
                for i, line in enumerate(_wrap_ja(comment)):
                    ja_texts.append((line, cx, base_y + i * 46, 34, (255, 255, 255), "ms"))

        _draw_ja_texts(frame, ja_texts)
        _draw_hud(frame, state)
        return frame
