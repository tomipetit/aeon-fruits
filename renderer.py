import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from game_state import GameState
from glass import draw_glass, preload as _preload_glass

# ---------- easing ----------

def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _ease_out_elastic(t: float) -> float:
    """Elastic ease-out — bounces several times before settling."""
    if t <= 0.0: return 0.0
    if t >= 1.0: return 1.0
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0


# ---------- backgrounds ----------

_bg_title: np.ndarray | None = None


def _get_bg_title() -> np.ndarray:
    global _bg_title
    if _bg_title is None:
        path = os.path.join(os.path.dirname(__file__), "assets", "title.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        _bg_title = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    return _bg_title


_bg_shop: np.ndarray | None = None


def _get_bg_shop() -> np.ndarray:
    global _bg_shop
    if _bg_shop is None:
        path = os.path.join(os.path.dirname(__file__), "assets", "shop.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        _bg_shop = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    return _bg_shop


_bg_order: np.ndarray | None = None


def _get_bg_order() -> np.ndarray:
    global _bg_order
    if _bg_order is None:
        path = os.path.join(os.path.dirname(__file__), "assets", "juice_stand_order.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        _bg_order = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    return _bg_order


_bg_pour: np.ndarray | None = None


def _get_bg_pour() -> np.ndarray:
    global _bg_pour
    if _bg_pour is None:
        path = os.path.join(os.path.dirname(__file__), "assets", "juice_stand_inside.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        _bg_pour = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    return _bg_pour


def _draw_bg(frame: np.ndarray, bg: np.ndarray) -> None:
    if bg.ndim == 3 and bg.shape[2] == 4:
        a = bg[:, :, 3:].astype(np.float32) / 255.0
        frame[:] = (bg[:, :, :3] * a + frame * (1.0 - a)).astype(np.uint8)
    else:
        frame[:] = bg[:, :, :3]


_animal_sprites: list[np.ndarray] | None = None  # 10 BGRA cells


def _get_animal_sprites() -> list[np.ndarray]:
    global _animal_sprites
    if _animal_sprites is None:
        sheet = cv2.imread(
            os.path.join(os.path.dirname(__file__), "assets", "animal_talk.png"),
            cv2.IMREAD_UNCHANGED,
        )
        cell_w = sheet.shape[1] // 10
        _animal_sprites = [sheet[:, i * cell_w:(i + 1) * cell_w] for i in range(10)]
    return _animal_sprites


# Y position of counter top in juice_stand_inside.png scaled to 1920×1080
_ANIMAL_COUNTER_TOP_Y = 860

_eval_sprites: list[list[np.ndarray]] | None = None  # [row][col], 4 rows x 10 cols


def _get_eval_sprites() -> list[list[np.ndarray]]:
    global _eval_sprites
    if _eval_sprites is None:
        sheet = cv2.imread(
            os.path.join(os.path.dirname(__file__), "assets", "animal_evaluation.png"),
            cv2.IMREAD_UNCHANGED,
        )
        cols, rows = 10, 4
        cw, rh = sheet.shape[1] // cols, sheet.shape[0] // rows
        _eval_sprites = [
            [sheet[r * rh:(r + 1) * rh, c * cw:(c + 1) * cw] for c in range(cols)]
            for r in range(rows)
        ]
    return _eval_sprites


def _eval_row(match_score: float) -> int:
    """Map match score to evaluation sprite row (0=best … 3=worst)."""
    if match_score >= 0.8: return 0
    if match_score >= 0.6: return 1
    if match_score >= 0.4: return 2
    return 3


_fruit_sprites: list[np.ndarray] | None = None  # 10 BGRA cells


def _get_fruit_sprites() -> list[np.ndarray]:
    global _fruit_sprites
    if _fruit_sprites is None:
        sheet = cv2.imread(
            os.path.join(os.path.dirname(__file__), "assets", "fruits.png"),
            cv2.IMREAD_UNCHANGED,
        )
        cell_w = sheet.shape[1] // 10
        _fruit_sprites = [sheet[:, i * cell_w:(i + 1) * cell_w] for i in range(10)]
    return _fruit_sprites


def _composite_sprite(frame: np.ndarray, sprite: np.ndarray, cx: int, cy: int, r: int) -> None:
    """Alpha-composite a square BGRA sprite centered at (cx, cy) with half-size r."""
    size = max(2, r * 2)
    s = cv2.resize(sprite, (size, size), interpolation=cv2.INTER_AREA)
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(frame.shape[1], cx + r), min(frame.shape[0], cy + r)
    sx1, sy1 = x1 - (cx - r), y1 - (cy - r)
    src = s[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]
    a = src[:, :, 3:].astype(np.float32) / 255.0
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = (src[:, :, :3] * a + roi * (1.0 - a)).astype(np.uint8)


def _composite_sprite_bottom(
    frame: np.ndarray, sprite: np.ndarray, cx: int, bottom_y: int, height: int
) -> None:
    """Alpha-composite BGRA sprite centered at cx with its bottom edge at bottom_y."""
    orig_h, orig_w = sprite.shape[:2]
    target_w = max(2, int(orig_w * height / orig_h))
    target_h = max(2, height)
    s = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)
    dx = cx - target_w // 2
    dy = bottom_y - target_h
    x1, y1 = max(0, dx), max(0, dy)
    x2, y2 = min(frame.shape[1], dx + target_w), min(frame.shape[0], dy + target_h)
    if x1 >= x2 or y1 >= y2:
        return
    src = s[y1 - dy:y1 - dy + (y2 - y1), x1 - dx:x1 - dx + (x2 - x1)]
    a = src[:, :, 3:].astype(np.float32) / 255.0
    frame[y1:y2, x1:x2] = (src[:, :, :3] * a + frame[y1:y2, x1:x2] * (1.0 - a)).astype(np.uint8)


_flash_src: np.ndarray | None = None


def _get_flash() -> np.ndarray:
    global _flash_src
    if _flash_src is None:
        path = os.path.join(os.path.dirname(__file__), "assets", "flash.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        _flash_src = cv2.resize(img, (config.WIDTH, config.HEIGHT), interpolation=cv2.INTER_AREA)
    return _flash_src


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

_FONT_JA_PATH      = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
_FONT_JA_BOLD_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
_font_ja_cache:      dict[int, ImageFont.FreeTypeFont] = {}
_font_ja_bold_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font_ja(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_ja_cache:
        _font_ja_cache[size] = ImageFont.truetype(_FONT_JA_PATH, size)
    return _font_ja_cache[size]


def _get_font_ja_bold(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_ja_bold_cache:
        _font_ja_bold_cache[size] = ImageFont.truetype(_FONT_JA_BOLD_PATH, size)
    return _font_ja_bold_cache[size]


def _draw_ja_texts(frame: np.ndarray,
                   texts: list[tuple[str, int, int, int, tuple, str]],
                   *,
                   bold: bool = False) -> None:
    """Batch-draw Japanese texts with a single BGR↔RGB conversion."""
    if not texts:
        return
    get_font = _get_font_ja_bold if bold else _get_font_ja
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, x, y, size, bgr, anchor in texts:
        rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
        draw.text((x, y), text, font=get_font(size), fill=rgb, anchor=anchor)
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

_INTRO1_TEXT = "森に新しくジュース屋さんができました。\n東三河で取れたおいしい果物を使ってジュースを作ってくれます。"
_INTRO2_TEXT  = "好みを聞いて、あなたに合うジュースをつくりますよ！"
_INTRO2_TEXT2 = "そこにいるみんなにはジュースづくりのお手伝いをしてもらおうかな！"

_STATE_LABELS = {
    GameState.IDLE:         "READY  [SPACE to start]",
    GameState.INTRO1:       "",
    GameState.INTRO2:       "",
    GameState.INTRO3:       "",
    GameState.ANIMAL:       "",
    GameState.FRUIT_SELECT: "入れたいフルーツのエリアに移動しよう！",
    GameState.MIX:          "みんなでジャンプしてミキサーを回そう！",
    GameState.POUR:         "",
    GameState.RESULT:       "",
}


def _alpha_blend(base: np.ndarray, overlay_bgr: np.ndarray, alpha: float) -> np.ndarray:
    cv2.addWeighted(overlay_bgr, alpha, base, 1.0 - alpha, 0, base)
    return base


def _draw_area_dividers(frame: np.ndarray):
    for i in range(1, config.NUM_AREAS):
        x = config.WIDTH * i // config.NUM_AREAS
        cv2.line(frame, (x, 0), (x, config.HEIGHT), (200, 200, 200), 3)


def _typewriter(text: str, elapsed: float, offset: float = 0.0) -> str:
    """Return the prefix of text visible at the given elapsed time."""
    chars = max(0, int((elapsed - offset) * config.TYPEWRITER_CHARS_PER_SEC))
    return text[:chars]


def _pixel_wrap(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    """Split text into lines so each line's pixel width fits within max_width."""
    lines: list[str] = []
    current = ""
    for ch in text:
        if ch == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = current + ch
        w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w > max_width and current:
            lines.append(current)
            current = ch
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def _draw_animal_bubble(
    frame: np.ndarray, data: dict, *, top_y: int | None = None,
    bottom_y: int | None = None, text: str | None = None, font_size: int = 44
) -> None:
    """Rounded speech bubble centered horizontally, max 90 % of screen width."""
    pref_text = text if text is not None else data["animal"]["pref"]
    font = _get_font_ja_bold(font_size)

    pad_x, pad_y = 48, 28
    max_bubble_w = int(config.WIDTH * 0.9)
    max_text_w = max_bubble_w - pad_x * 2

    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = _pixel_wrap(pref_text, font, max_text_w, dummy_draw)

    line_bboxes = [dummy_draw.textbbox((0, 0), ln, font=font) for ln in lines]
    max_tw = max(bb[2] - bb[0] for bb in line_bboxes)
    line_h = max(bb[3] - bb[1] for bb in line_bboxes)
    line_spacing = 20
    total_text_h = line_h * len(lines) + line_spacing * (len(lines) - 1)

    bw = min(max_tw + pad_x * 2, max_bubble_w)
    bh = total_text_h + pad_y * 2
    cx = config.WIDTH // 2
    bx1, bx2 = cx - bw // 2, cx + bw // 2

    if top_y is not None:
        by1 = top_y
        by2 = by1 + bh
    elif bottom_y is not None:
        by2 = bottom_y
        by1 = by2 - bh
    else:
        animal_top = _ANIMAL_COUNTER_TOP_Y - 500
        by2 = animal_top - 16
        by1 = by2 - bh

    _draw_rounded_rect(frame, (bx1, by1), (bx2, by2), (30, 30, 30),    radius=28, thickness=-1)
    _draw_rounded_rect(frame, (bx1, by1), (bx2, by2), (255, 255, 255), radius=28, thickness=4)

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    text_y_start = by1 + pad_y + line_h // 2
    for i, line in enumerate(lines):
        draw.text(
            (cx, text_y_start + i * (line_h + line_spacing)),
            line, font=font, fill=(255, 255, 255), anchor="mm",
        )
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)



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
    lw, lh, radius = 300, 72, 22
    ly = config.HEIGHT - 100
    ja_texts = []
    for i, fruit in enumerate(data["area_fruits"]):
        cx = i * area_w + area_w // 2
        x1, x2 = cx - lw // 2, cx + lw // 2
        color = fruit["bgr"]
        _draw_rounded_rect(frame, (x1, ly), (x2, ly + lh), color,  radius=radius, thickness=-1)
        _draw_rounded_rect(frame, (x1, ly), (x2, ly + lh), (255, 255, 255), radius=radius, thickness=3)
        ja_texts.append((fruit["name"], cx, ly + lh // 2, 38, (255, 255, 255), "mm"))
    return ja_texts


def _blended_juice_color(data: dict) -> tuple[int, int, int]:
    blended = np.zeros(3, dtype=np.float32)
    for i, prop in enumerate(data["fruit_proportions"]):
        blended += np.array(data["area_fruits"][i]["bgr"], dtype=np.float32) * prop
    return tuple(int(c) for c in blended)


def _draw_rounded_rect(img: np.ndarray, pt1, pt2, color, radius: int, thickness: int = -1) -> None:
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0,   0,  90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0,  90, 180, color, thickness, cv2.LINE_AA)


def _draw_countdown(frame: np.ndarray, elapsed: float) -> None:
    duration = config.PHASE_DURATIONS["FRUIT_SELECT"]
    remaining = duration - elapsed
    if remaining > 10:
        return

    count = max(1, math.ceil(remaining))

    # Pulse: scale up briefly each time the digit changes
    time_since_change = (1.0 - (remaining % 1.0)) % 1.0
    pulse = 1.0 + 0.3 * max(0.0, 1.0 - time_since_change / 0.25)

    # Urgent color for last 3 seconds
    color = (80, 80, 255) if count <= 3 else (255, 255, 255)

    cx, cy = config.WIDTH // 2, config.HEIGHT // 2
    text = str(count)
    scale = 14.0 * pulse
    thickness = 22

    (tw, th), _ = cv2.getTextSize(text, _FONT_BOLD, scale, thickness)
    tx, ty = cx - tw // 2, cy + th // 2

    overlay = frame.copy()
    cv2.putText(overlay, text, (tx + 10, ty + 10), _FONT_BOLD, scale, (0, 0, 0), thickness + 10, cv2.LINE_AA)
    cv2.putText(overlay, text, (tx, ty), _FONT_BOLD, scale, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


def _draw_mixer(frame: np.ndarray, data: dict):
    juice_color = _blended_juice_color(data)
    mix_level = data["mix_level"]
    cx, cy, r = config.WIDTH // 2, config.HEIGHT // 2, 360

    # Background circle
    cv2.circle(frame, (cx, cy), r, (80, 80, 80), -1)
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 6)

    # Juice circle expanding from center
    juice_r = max(0, int(r * mix_level))
    if juice_r > 0:
        cv2.circle(frame, (cx, cy), juice_r, juice_color, -1)

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
    y = _ANIMAL_COUNTER_TOP_Y + 55

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
            cv2.polylines(frame, [pts_arr], True, (255, 255, 255), 2)
        elif fill >= 0.5:
            # Half star — white base, gold left half
            cv2.fillPoly(frame, [pts_arr], (255, 255, 255))
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts_arr], 255)
            mask[:, cx:] = 0
            frame[mask > 0] = (0, 215, 255)
            cv2.polylines(frame, [pts_arr], True, (255, 255, 255), 2)
        else:
            # Empty star — white fill with white outline
            cv2.fillPoly(frame, [pts_arr], (255, 255, 255))
            cv2.polylines(frame, [pts_arr], True, (255, 255, 255), 2)

    score_pct = int(data["match_score"] * 100)
    score_text = f"MATCH {score_pct}%"
    (tw, _), _ = cv2.getTextSize(score_text, _FONT_BOLD, 2.0, 4)
    tx = config.WIDTH // 2 - tw // 2
    ty = _ANIMAL_COUNTER_TOP_Y + 155
    cv2.putText(frame, score_text, (tx, ty), _FONT_BOLD, 2.0, (10, 45, 100), 12, cv2.LINE_AA)
    cv2.putText(frame, score_text, (tx, ty), _FONT_BOLD, 2.0, (255, 255, 255),  4, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, state: GameState):
    label = _STATE_LABELS.get(state, "")
    if not label:
        return
    if state in (GameState.IDLE, GameState.FRUIT_SELECT, GameState.MIX) and int(time.time() * 2) % 2 == 0:
        return
    if state == GameState.IDLE:
        y = config.HEIGHT - 60
    elif state == GameState.FRUIT_SELECT:
        y = 160
    elif state == GameState.MIX:
        y = 100
    else:
        y = 170
    _draw_ja_texts(frame, [(label, config.WIDTH // 2, y, 40, (255, 255, 255), "mm")], bold=True)


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
        # Preload assets so first frame has no I/O latency
        _get_bg_title()
        _get_bg_shop()
        _get_bg_order()
        _get_bg_pour()
        _get_flash()
        _get_fruit_sprites()
        _get_animal_sprites()
        _get_eval_sprites()
        _preload_glass()

    def _reset(self):
        self._smooth_radii = [_ICON_RADIUS_MIN] * config.NUM_AREAS
        self._smooth_fill_levels = [1.0 / config.NUM_AREAS] * config.NUM_AREAS

    def _draw_area_bottles(self, frame: np.ndarray, data: dict):
        area_w = config.WIDTH // config.NUM_AREAS
        bottle_top = 190
        bottle_bottom = config.HEIGHT - 115
        bottle_h = bottle_bottom - bottle_top
        radius = 20
        for i, fruit in enumerate(data["area_fruits"]):
            x1 = i * area_w + 78
            x2 = (i + 1) * area_w - 78
            fill_h = int(bottle_h * self._smooth_fill_levels[i])
            fill_y = bottle_bottom - fill_h
            color = fruit["bgr"]

            # Clip fill to rounded bottle shape
            bottle_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            _draw_rounded_rect(bottle_mask, (x1, bottle_top), (x2, bottle_bottom), 255, radius=radius, thickness=-1)
            fill_overlay = frame.copy()
            cv2.rectangle(fill_overlay, (x1, fill_y), (x2, bottle_bottom), color, -1)
            blended = cv2.addWeighted(fill_overlay, 0.34, frame, 0.66, 0)
            frame[bottle_mask > 0] = blended[bottle_mask > 0]

            # Rounded outline (neck removed)
            _draw_rounded_rect(frame, (x1, bottle_top), (x2, bottle_bottom),
                               (255, 255, 255), radius=radius, thickness=4)
            cv2.line(frame, (x1 + 16, fill_y), (x2 - 16, fill_y),
                     (255, 255, 255), 2, cv2.LINE_AA)

    def render(self, bgr_frame: np.ndarray, data: dict) -> np.ndarray:
        state: GameState = data["state"]

        # Reset particles at the start of each new round
        if self._prev_state != state and state == GameState.ANIMAL:
            self._reset()
        self._prev_state = state

        frame = bgr_frame.copy()
        if state != GameState.MIX:
            _draw_area_dividers(frame)

        ja_texts: list = []

        if state == GameState.IDLE:
            _draw_bg(frame, _get_bg_title())

        elif state == GameState.INTRO1:
            _draw_bg(frame, _get_bg_shop())
            _draw_animal_bubble(frame, data, bottom_y=config.HEIGHT - 40,
                                text=_typewriter(_INTRO1_TEXT, data["elapsed"]), font_size=38)

        elif state == GameState.INTRO2:
            _draw_bg(frame, _get_bg_order())
            if data["elapsed"] >= 4.0:
                intro2_text = _typewriter(_INTRO2_TEXT2, data["elapsed"], offset=4.0)
            else:
                intro2_text = _typewriter(_INTRO2_TEXT, data["elapsed"])
            _draw_animal_bubble(frame, data, bottom_y=config.HEIGHT - 40, text=intro2_text, font_size=38)

        elif state == GameState.INTRO3:
            # Semi-transparent black overlay
            black = np.zeros_like(frame)
            cv2.addWeighted(black, 0.75, frame, 0.25, 0, frame)
            # Large "GAME START"
            gs = "GAME START"
            (tw, th), _ = cv2.getTextSize(gs, _FONT_BOLD, 6.0, 16)
            tx, ty = config.WIDTH // 2 - tw // 2, config.HEIGHT // 2 - 40
            cv2.putText(frame, gs, (tx, ty), _FONT_BOLD, 6.0, (0, 0, 0),     24, cv2.LINE_AA)
            cv2.putText(frame, gs, (tx, ty), _FONT_BOLD, 6.0, (255, 255, 255), 16, cv2.LINE_AA)
            # Japanese subtitle
            ja_texts.append(("どうぶつの好みを聞いて、それに合った果物を選んでみよう",
                             config.WIDTH // 2, config.HEIGHT // 2 + th + 20, 42, (255, 255, 255), "mm"))

        elif state == GameState.ANIMAL:
            _draw_bg(frame, _get_bg_pour())
            sprites = _get_animal_sprites()
            sprite_idx = config.ANIMAL_SPRITE_INDEX.get(data["animal"]["name"], 0)
            _composite_sprite_bottom(
                frame, sprites[sprite_idx], config.WIDTH // 2, _ANIMAL_COUNTER_TOP_Y, 500
            )
            if data["elapsed"] >= 1.0:
                pref = data["animal"]["pref"]
                _draw_animal_bubble(frame, data,
                                    text=_typewriter(pref, data["elapsed"], offset=1.0))

        elif state == GameState.FRUIT_SELECT:
            # Update smooth radii toward target from live fruit_proportions
            for i, prop in enumerate(data["fruit_proportions"]):
                target = _ICON_RADIUS_MIN + (_ICON_RADIUS_MAX - _ICON_RADIUS_MIN) * prop
                self._smooth_radii[i] += (target - self._smooth_radii[i]) * _ICON_LERP
                self._smooth_fill_levels[i] += (prop - self._smooth_fill_levels[i]) * _BOTTLE_LERP

            self._draw_area_bottles(frame, data)
            # Draw centered fruit icons sized by proportion
            area_w = config.WIDTH // config.NUM_AREAS
            sprites = _get_fruit_sprites()
            for i, fruit in enumerate(data["area_fruits"]):
                cx = i * area_w + area_w // 2
                cy = config.HEIGHT // 2
                r = max(2, int(self._smooth_radii[i]))
                sprite_idx = config.FRUIT_SPRITE_INDEX.get(fruit["name"], 0)
                _composite_sprite(frame, sprites[sprite_idx], cx, cy, r)

            ja_texts += _draw_fruit_labels(frame, data)
            pref = data["animal"]["pref"]
            _draw_animal_bubble(frame, data, top_y=20,
                                text=_typewriter(pref, data["elapsed"]))
            _draw_countdown(frame, data["elapsed"])

        elif state == GameState.MIX:
            _draw_mixer(frame, data)

        elif state == GameState.POUR:
            juice_color = _blended_juice_color(data)
            elapsed = data["elapsed"]

            # Background
            _draw_bg(frame, _get_bg_pour())

            # Flash (between background and glass)
            _draw_bg(frame, _get_flash())

            # Pop scale: elastic ease-out, settles at 1.0 s
            pop_t = _ease_out_elastic(min(1.0, elapsed / 1.0))
            scale = max(0.05, pop_t)

            # Fill: ease-out-cubic, completes at 1 s
            fill = _ease_out_cubic(min(1.0, elapsed / 1.0))

            gw = max(4, int(500 * scale))
            gh = max(4, int(700 * scale))
            gx = (config.WIDTH  - gw) // 2
            gy = (config.HEIGHT - gh) // 2
            draw_glass(frame, gx, gy, gw, gh, color_bgr=juice_color, fill_level=fill)
            _draw_animal_bubble(frame, data, top_y=60,
                                text=_typewriter("完成！", data["elapsed"]))

        elif state == GameState.RESULT:
            _draw_bg(frame, _get_bg_pour())
            # Animal evaluation sprite
            col = config.ANIMAL_SPRITE_INDEX.get(data["animal"]["name"], 0)
            row = _eval_row(data["match_score"])
            _composite_sprite_bottom(
                frame, _get_eval_sprites()[row][col], config.WIDTH // 2, _ANIMAL_COUNTER_TOP_Y, 500
            )
            # Taste comment bubble above the animal
            comment = data.get("taste_comment", "")
            if comment:
                _draw_animal_bubble(frame, data,
                                    text=_typewriter(comment, data["elapsed"]), font_size=36)
            _draw_stars(frame, data)

        _draw_ja_texts(frame, ja_texts)
        _draw_hud(frame, state)
        return frame
