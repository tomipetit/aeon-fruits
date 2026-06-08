import cv2
import numpy as np

import config
from game_state import GameState


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# Japanese text cannot be rendered with OpenCV's built-in font.
# These placeholder ASCII labels will be replaced once Pillow/ImageFont
# support is wired in with proper Japanese font files.
_STATE_LABELS = {
    GameState.IDLE:   "READY  [SPACE to start]",
    GameState.ANIMAL: "",
    GameState.AREA:   "CHOOSE YOUR FRUIT AREA!",
    GameState.FILL:   "JUMP to fill your area!",
    GameState.MIX:    "SPIN to mix!",
    GameState.RESULT: "",
}


def _alpha_blend(base: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Blend overlay_bgr onto base in-place and return base."""
    cv2.addWeighted(overlay_bgr, alpha, base, 1.0 - alpha, 0, base)
    return base


def _draw_area_dividers(frame: np.ndarray):
    for i in range(1, config.NUM_AREAS):
        x = config.WIDTH * i // config.NUM_AREAS
        cv2.line(frame, (x, 0), (x, config.HEIGHT), (200, 200, 200), 3)


def _draw_fill_overlay(frame: np.ndarray, data: dict):
    area_w = config.WIDTH // config.NUM_AREAS
    for i, level in enumerate(data["fill_levels"]):
        if level <= 0:
            continue
        fruit = data["area_fruits"][i]
        color = fruit["bgr"]
        x1 = i * area_w
        x2 = x1 + area_w
        fill_h = int(config.HEIGHT * level)
        y1 = config.HEIGHT - fill_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, config.HEIGHT), color, -1)
        _alpha_blend(frame, overlay, config.FILL_OVERLAY_ALPHA)

        # Fruit name label (ASCII placeholder)
        label = fruit["name"]
        cv2.putText(frame, label, (x1 + 20, config.HEIGHT - 30),
                    _FONT, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


def _draw_area_counts(frame: np.ndarray, data: dict):
    area_w = config.WIDTH // config.NUM_AREAS
    for i, count in enumerate(data["area_counts"]):
        x = i * area_w + area_w // 2 - 30
        cv2.putText(frame, str(count), (x, 80), _FONT_BOLD, 3.0,
                    (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, str(count), (x, 80), _FONT_BOLD, 3.0,
                    (0, 0, 0), 2, cv2.LINE_AA)


def _draw_animal_box(frame: np.ndarray, data: dict):
    animal = data["animal"]
    box_x, box_y = 60, 120
    cv2.rectangle(frame, (box_x, box_y), (box_x + 700, box_y + 160),
                  (30, 30, 30), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + 700, box_y + 160),
                  (255, 255, 255), 3)
    cv2.putText(frame, animal["name"], (box_x + 20, box_y + 70),
                _FONT_BOLD, 2.5, (255, 220, 100), 4, cv2.LINE_AA)
    # Preference text (ASCII fallback)
    pref = animal["pref"]
    cv2.putText(frame, pref, (box_x + 20, box_y + 140),
                _FONT, 1.2, (220, 220, 220), 2, cv2.LINE_AA)


def _draw_stars(frame: np.ndarray, data: dict):
    stars = data["star_rating"]
    star_size = 60
    total_w = stars * (star_size + 10)
    start_x = (config.WIDTH - total_w) // 2
    y = config.HEIGHT // 2

    for i in range(stars):
        cx = start_x + i * (star_size + 10) + star_size // 2
        pts = []
        import math
        for j in range(5):
            outer_angle = math.pi / 2 + j * 2 * math.pi / 5
            inner_angle = outer_angle + math.pi / 5
            pts.append([int(cx + star_size * math.cos(outer_angle)),
                        int(y - star_size * math.sin(outer_angle))])
            pts.append([int(cx + star_size * 0.4 * math.cos(inner_angle)),
                        int(y - star_size * 0.4 * math.sin(inner_angle))])
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts_arr], (0, 215, 255))
        cv2.polylines(frame, [pts_arr], True, (0, 165, 200), 2)

    score_pct = int(data["match_score"] * 100)
    cv2.putText(frame, f"MATCH {score_pct}%", (config.WIDTH // 2 - 120, y + 100),
                _FONT_BOLD, 2.0, (255, 255, 255), 4, cv2.LINE_AA)


def _draw_mixer(frame: np.ndarray, data: dict):
    """Draw a simple mixer circle with blended juice color."""
    fill_levels = data["fill_levels"]
    area_fruits = data["area_fruits"]
    total = sum(fill_levels) or 1.0
    props = [lvl / total for lvl in fill_levels]

    blended = np.zeros(3, dtype=np.float32)
    for i, prop in enumerate(props):
        color = np.array(area_fruits[i]["bgr"], dtype=np.float32)
        blended += color * prop
    bgr = tuple(int(c) for c in blended)

    cx, cy, r = config.WIDTH // 2, config.HEIGHT // 2, 200
    cv2.circle(frame, (cx, cy), r, bgr, -1)
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 4)
    cv2.putText(frame, "MIXING!", (cx - 90, cy + 12),
                _FONT_BOLD, 1.8, (255, 255, 255), 4, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, state_label: str):
    if not state_label:
        return
    cv2.putText(frame, state_label,
                (config.WIDTH // 2 - len(state_label) * 14, config.HEIGHT - 40),
                _FONT, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


class ARRenderer:
    def render(self, bgr_frame: np.ndarray, data: dict) -> np.ndarray:
        frame = bgr_frame.copy()
        state: GameState = data["state"]

        _draw_area_dividers(frame)

        if state == GameState.IDLE:
            pass

        elif state == GameState.ANIMAL:
            _draw_animal_box(frame, data)

        elif state == GameState.AREA:
            _draw_area_counts(frame, data)
            _draw_animal_box(frame, data)

        elif state == GameState.FILL:
            _draw_fill_overlay(frame, data)
            _draw_area_counts(frame, data)

        elif state == GameState.MIX:
            _draw_fill_overlay(frame, data)
            _draw_mixer(frame, data)

        elif state == GameState.RESULT:
            _draw_fill_overlay(frame, data)
            _draw_animal_box(frame, data)
            _draw_stars(frame, data)

        _draw_hud(frame, _STATE_LABELS.get(state, ""))
        return frame
