"""
山のジュース屋さん ARゲーム

Usage:
    python main.py               # NDI interactive source selection
    python main.py --demo        # webcam demo (no NDI required)
    python main.py --no-camera   # camera-less demo: z/x/c keys adjust area proportions
    python main.py --debug       # show detection overlay

Keys during runtime:
    SPACE   start a round (from IDLE)
    d       toggle debug overlay
    q / ESC quit
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Load ANTHROPIC_API_KEY from .env if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import config
from detector import MotionDetector
from game_state import JuiceGame
from renderer import ARRenderer


def _shift_proportion(props: list[float], idx: int, step: float = 0.05) -> list[float]:
    props = props[:]
    props[idx] = min(1.0, props[idx] + step)
    others_sum = sum(p for j, p in enumerate(props) if j != idx)
    target_others = 1.0 - props[idx]
    if others_sum > 0:
        for j in range(len(props)):
            if j != idx:
                props[j] = props[j] / others_sum * target_others
    return props


def _make_blank_frame() -> np.ndarray:
    return np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use webcam instead of NDI")
    parser.add_argument("--no-camera", action="store_true", help="camera-less demo: z/x/c keys adjust area proportions")
    parser.add_argument("--debug", action="store_true", help="show detection overlay")
    args = parser.parse_args()

    # ----- Input source -----
    if args.no_camera:
        proportions = [1.0 / config.NUM_AREAS] * config.NUM_AREAS
        print("カメラなしデモモード: z/x/c キーで配合割合を調整できます")
        cap = None
        receiver = None
    elif args.demo:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)
        print("デモモード: Webカメラ使用")
        receiver = None
    else:
        from ndi_receiver import NDIReceiver
        receiver = NDIReceiver()
        if config.NDI_SOURCE_INDEX is not None:
            name = receiver.connect_by_index(config.NDI_SOURCE_INDEX)
        else:
            name = receiver.connect_interactive()
        print(f"NDI接続: {name}")
        cap = None

    # ----- Window -----
    win = "AR Game"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if config.DISPLAY_MONITOR_OFFSET_X:
        cv2.moveWindow(win, config.DISPLAY_MONITOR_OFFSET_X, 0)

    # ----- Subsystems -----
    detector = MotionDetector()
    if args.no_camera:
        game = JuiceGame(demo_mode=True, demo_proportions=proportions)
    else:
        game = JuiceGame()
    renderer = ARRenderer()
    debug_on = args.debug

    prev_time = time.monotonic()
    frame_count = 0

    print("起動完了。SPACEキーでゲーム開始。")

    while True:
        # -- Capture --
        if args.no_camera:
            bgr = _make_blank_frame()
        elif receiver is not None:
            bgr = receiver.get_frame()
            if bgr is None:
                bgr = _make_blank_frame()
        else:
            ret, bgr = cap.read()
            if not ret:
                bgr = _make_blank_frame()
            else:
                bgr = cv2.resize(bgr, (config.WIDTH, config.HEIGHT))

        # Mirror flip: project screen faces children, so flip horizontally
        bgr = cv2.flip(bgr, 1)

        # -- Detection --
        detector.update(bgr)
        area_counts = detector.get_area_counts()
        jump_counts = detector.get_jump_counts()
        spinning = detector.is_spinning()

        # -- Game logic --
        game.update(area_counts, jump_counts, spinning)
        render_data = game.get_render_data()

        # -- AR render --
        if debug_on:
            display = detector.get_debug_overlay(bgr)
        else:
            display = renderer.render(bgr, render_data)

        # -- FPS overlay --
        frame_count += 1
        now = time.monotonic()
        if now - prev_time >= 1.0:
            fps = frame_count / (now - prev_time)
            frame_count = 0
            prev_time = now
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(win, display)

        # -- Keys --
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):   # ESC or q
            break
        elif key == ord(" "):
            game.start()
            print("ゲーム開始!")
        elif key == ord("d"):
            debug_on = not debug_on
            print(f"デバッグ表示: {'ON' if debug_on else 'OFF'}")
        elif args.no_camera and key in (ord("z"), ord("x"), ord("c")):
            area_index = {ord("z"): 0, ord("x"): 1, ord("c"): 2}[key]
            proportions = _shift_proportion(game.demo_proportions, area_index)
            game.set_demo_proportions(proportions)
            print(f"配合: {' | '.join(f'エリア{i+1}: {p*100:.0f}%' for i, p in enumerate(proportions))}")

    # -- Cleanup --
    cv2.destroyAllWindows()
    if receiver is not None:
        receiver.close()
    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
