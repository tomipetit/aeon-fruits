"""
山のジュース屋さん ARゲーム

Usage:
    python main.py               # NDI interactive source selection
    python main.py --demo        # webcam demo (no NDI required)
    python main.py --debug       # show detection overlay

Keys during runtime:
    SPACE   start a round (from IDLE)
    d       toggle debug overlay
    q / ESC quit
"""

import argparse
import sys
import time

import cv2
import numpy as np

import config
from detector import MotionDetector
from game_state import JuiceGame
from renderer import ARRenderer


def _make_blank_frame() -> np.ndarray:
    return np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use webcam instead of NDI")
    parser.add_argument("--debug", action="store_true", help="show detection overlay")
    args = parser.parse_args()

    # ----- Input source -----
    if args.demo:
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
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if config.DISPLAY_MONITOR_OFFSET_X:
        cv2.moveWindow(win, config.DISPLAY_MONITOR_OFFSET_X, 0)

    # ----- Subsystems -----
    detector = MotionDetector()
    game = JuiceGame()
    renderer = ARRenderer()
    debug_on = args.debug

    prev_time = time.monotonic()
    frame_count = 0

    print("起動完了。SPACEキーでゲーム開始。")

    while True:
        # -- Capture --
        if receiver is not None:
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

    # -- Cleanup --
    cv2.destroyAllWindows()
    if receiver is not None:
        receiver.close()
    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
