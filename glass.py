"""
Renders a glass with freely configurable liquid color.

Layering order (bottom to top):
  1. frame background
  2. liquid — solid color shaped by liquid_mask.png alpha
  3. glass overlay — glass.png composited with its own alpha
"""

import os

import cv2
import numpy as np

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")

_glass_src: np.ndarray | None = None   # BGRA, original size
_mask_src: np.ndarray | None = None    # BGRA, original size


def _load():
    global _glass_src, _mask_src
    if _glass_src is None:
        _glass_src = cv2.imread(os.path.join(_ASSETS, "glass.png"), cv2.IMREAD_UNCHANGED)
    if _mask_src is None:
        _mask_src = cv2.imread(os.path.join(_ASSETS, "liquid_mask.png"), cv2.IMREAD_UNCHANGED)


def preload() -> None:
    """Eagerly load assets so the first draw_glass call has no I/O latency."""
    _load()


def _alpha_composite(dst: np.ndarray, src_bgr: np.ndarray, src_alpha: np.ndarray) -> None:
    """In-place alpha-composite src over dst (all same HxW)."""
    a = src_alpha.astype(np.float32) / 255.0
    a3 = a[:, :, np.newaxis]
    dst[:] = (src_bgr * a3 + dst * (1.0 - a3)).astype(np.uint8)


def draw_glass(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color_bgr: tuple[int, int, int],
    *,
    fill_level: float = 1.0,
) -> None:
    """
    Draw a glass containing liquid of the given color onto *frame*.

    Parameters
    ----------
    frame      : BGR frame to draw on (modified in-place)
    x, y       : top-left corner of the glass in frame coordinates
    width, height : render size in pixels
    color_bgr  : drink color as (B, G, R)
    fill_level : 0.0 (empty) … 1.0 (full), fills from the bottom up
    """
    _load()

    # ---- resize assets to requested dimensions ----
    glass = cv2.resize(_glass_src, (width, height), interpolation=cv2.INTER_AREA)
    mask  = cv2.resize(_mask_src,  (width, height), interpolation=cv2.INTER_AREA)

    glass_bgr   = glass[:, :, :3]
    glass_alpha = glass[:, :, 3]
    mask_alpha  = mask[:, :, 3]

    # ---- apply fill level (cut mask from top) ----
    if fill_level < 1.0:
        cutoff = int(height * (1.0 - max(0.0, min(1.0, fill_level))))
        mask_alpha = mask_alpha.copy()
        mask_alpha[:cutoff, :] = 0

    # ---- clip destination rect to frame bounds ----
    fh, fw = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + width, fw), min(y + height, fh)
    if x1 >= x2 or y1 >= y2:
        return

    # offsets into the resized asset arrays
    ox, oy = x1 - x, y1 - y

    roi         = frame[y1:y2, x1:x2]
    liq_alpha   = mask_alpha[oy:oy + (y2 - y1), ox:ox + (x2 - x1)]
    g_bgr       = glass_bgr [oy:oy + (y2 - y1), ox:ox + (x2 - x1)]
    g_alpha     = glass_alpha[oy:oy + (y2 - y1), ox:ox + (x2 - x1)]

    # ---- 1. composite liquid color ----
    liquid = np.empty_like(roi)
    liquid[:] = color_bgr
    _alpha_composite(roi, liquid, liq_alpha)

    # ---- 2. composite glass on top ----
    _alpha_composite(roi, g_bgr, g_alpha)


if __name__ == "__main__":
    WIN = "glass preview  [q: quit]"
    W, H = 400, 560
    PAD = 20

    cv2.namedWindow(WIN)
    cv2.createTrackbar("R",    WIN, 255, 255, lambda _: None)
    cv2.createTrackbar("G",    WIN, 120, 255, lambda _: None)
    cv2.createTrackbar("B",    WIN,   0, 255, lambda _: None)
    cv2.createTrackbar("Fill %", WIN, 100, 100, lambda _: None)

    while True:
        r = cv2.getTrackbarPos("R",      WIN)
        g = cv2.getTrackbarPos("G",      WIN)
        b = cv2.getTrackbarPos("B",      WIN)
        fill = cv2.getTrackbarPos("Fill %", WIN) / 100.0

        canvas = np.full((H + PAD * 2, W + PAD * 2, 3), 50, dtype=np.uint8)
        draw_glass(canvas, PAD, PAD, W, H, color_bgr=(b, g, r), fill_level=fill)

        # show current RGB hex in corner
        hex_label = f"#{r:02X}{g:02X}{b:02X}  fill={int(fill*100)}%"
        cv2.putText(canvas, hex_label, (PAD, H + PAD * 2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WIN, canvas)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
