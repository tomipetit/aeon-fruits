import math

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


def _proportions_to_units(proportions: list[float], total_units: int = 30) -> list[int]:
    raw = [max(0.0, p) * total_units for p in proportions]
    units = [int(v) for v in raw]
    remaining = total_units - sum(units)
    remainders = sorted(
        range(len(raw)),
        key=lambda i: raw[i] - units[i],
        reverse=True,
    )
    for i in remainders[:max(0, remaining)]:
        units[i] += 1
    return units


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
    stars = data["star_rating"]  # float, 0.5 steps, 0.5–5.0
    star_size = 60
    n_stars = 5
    total_w = n_stars * (star_size * 2 + 10)
    start_x = (config.WIDTH - total_w) // 2
    y = config.HEIGHT // 2 + 160

    for i in range(n_stars):
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

        fill = max(0.0, min(1.0, stars - i))

        if fill >= 1.0:
            # Full star
            cv2.fillPoly(frame, [pts_arr], (0, 215, 255))
            cv2.polylines(frame, [pts_arr], True, (0, 165, 200), 2)
        elif fill >= 0.5:
            # Half star — fill left half using mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts_arr], 255)
            mask[:, cx:] = 0
            frame[mask > 0] = (0, 215, 255)
            cv2.polylines(frame, [pts_arr], True, (0, 165, 200), 2)
        else:
            # Empty star — outline only
            cv2.polylines(frame, [pts_arr], True, (120, 120, 120), 2)

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
_BOTTLE_LERP = 0.12
_MIX_TOTAL_UNITS = 30


class ARRenderer:
    def __init__(self):
        self._smooth_radii: list[float] = [_ICON_RADIUS_MIN] * config.NUM_AREAS
        self._smooth_fill_levels: list[float] = [1.0 / config.NUM_AREAS] * config.NUM_AREAS
        self._prev_state: GameState | None = None

    def _reset(self):
        self._smooth_radii = [_ICON_RADIUS_MIN] * config.NUM_AREAS
        self._smooth_fill_levels = [1.0 / config.NUM_AREAS] * config.NUM_AREAS

    def _draw_area_bottles(self, frame: np.ndarray, data: dict):
        area_w = config.WIDTH // config.NUM_AREAS
        bottle_top = 190
        bottle_bottom = config.HEIGHT - 115
        bottle_h = bottle_bottom - bottle_top
        for i, fruit in enumerate(data["area_fruits"]):
            x1 = i * area_w + 78
            x2 = (i + 1) * area_w - 78
            fill_h = int(bottle_h * self._smooth_fill_levels[i])
            fill_y = bottle_bottom - fill_h
            color = fruit["bgr"]

            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, fill_y), (x2, bottle_bottom), color, -1)
            _alpha_blend(frame, overlay, 0.34)

            neck_w = max(90, (x2 - x1) // 4)
            neck_x1 = (x1 + x2 - neck_w) // 2
            neck_x2 = neck_x1 + neck_w
            cv2.rectangle(frame, (neck_x1, bottle_top - 36), (neck_x2, bottle_top),
                          (255, 255, 255), 3)
            cv2.rectangle(frame, (x1, bottle_top), (x2, bottle_bottom),
                          (255, 255, 255), 4)
            cv2.line(frame, (x1 + 16, fill_y), (x2 - 16, fill_y),
                     (255, 255, 255), 2, cv2.LINE_AA)

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
            # Update smooth radii toward target from live fruit_proportions
            for i, prop in enumerate(data["fruit_proportions"]):
                target = _ICON_RADIUS_MIN + (_ICON_RADIUS_MAX - _ICON_RADIUS_MIN) * prop
                self._smooth_radii[i] += (target - self._smooth_radii[i]) * _ICON_LERP
                self._smooth_fill_levels[i] += (prop - self._smooth_fill_levels[i]) * _BOTTLE_LERP

            self._draw_area_bottles(frame, data)
            mix_units = _proportions_to_units(data["fruit_proportions"], _MIX_TOTAL_UNITS)
            # Draw centered fruit icons sized by proportion
            area_w = config.WIDTH // config.NUM_AREAS
            for i, fruit in enumerate(data["area_fruits"]):
                cx = i * area_w + area_w // 2
                cy = config.HEIGHT // 2
                r = max(2, int(self._smooth_radii[i]))
                color = fruit["bgr"]
                cv2.circle(frame, (cx, cy), r, color, -1)
                cv2.circle(frame, (cx, cy), r, (255, 255, 255), max(3, r // 25))
                # Show mix units inside icon (sum is 30)
                if r > 40:
                    ratio_str = str(mix_units[i])
                    fscale = max(1.2, r / 90.0) * 2.0
                    thickness = max(4, r // 30)
                    (tw, th), _ = cv2.getTextSize(ratio_str, _FONT_BOLD, fscale, thickness)
                    tx, ty = cx - tw // 2, cy + th // 2
                    cv2.putText(frame, ratio_str, (tx, ty), _FONT_BOLD, fscale,
                                (255, 255, 255), thickness + 4, cv2.LINE_AA)
                    cv2.putText(frame, ratio_str, (tx, ty), _FONT_BOLD, fscale,
                                (30, 30, 30), thickness, cv2.LINE_AA)

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
                    ja_texts.append((line, cx, base_y + i * 70, 34, (255, 255, 255), "mm"))

        _draw_ja_texts(frame, ja_texts)
        _draw_hud(frame, state)
        return frame
