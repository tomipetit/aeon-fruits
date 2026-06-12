from collections import deque
import contextlib
import os
import threading
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
from ultralytics import YOLO

import config


@contextlib.contextmanager
def _mute_stderr():
    """Suppress C-level stderr (e.g. MPS fallback warnings from PyTorch)."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class _TrackedBlob:
    _next_id = 0

    def __init__(self, cx: float, cy: float, area_index: int):
        self.blob_id = _TrackedBlob._next_id
        _TrackedBlob._next_id += 1
        self.cx = cx
        self.cy = cy
        self.area_index = area_index
        self.y_history: deque[float] = deque(maxlen=45)
        self.y_history.append(cy)
        self.last_seen_frame = 0
        self._jump_state = "GROUND"   # GROUND | RISING | FALLING
        self._peak_y = cy
        self._last_jump_time = 0.0
        self.bbox: tuple | None = None  # (x1, y1, x2, y2) in scaled coords


class MotionDetector:
    def __init__(self):
        self._model_person = YOLO(config.YOLO_MODEL_PERSON)
        self._model_face   = YOLO(config.YOLO_MODEL_FACE)
        self._model_person.to("mps")
        self._model_face.to("mps")

        self._scale = config.DETECTION_SCALE
        self._area_w = int(config.WIDTH * self._scale / config.NUM_AREAS)
        self._frame_h_scaled = int(config.HEIGHT * self._scale)
        self._frame_w_scaled = int(config.WIDTH * self._scale)
        self._dead_zone_px = int(config.WIDTH * self._scale * config.AREA_DEAD_ZONE)

        self._blobs: list[_TrackedBlob] = []
        self._frame_number = 0
        self._jump_counts = [0] * config.NUM_AREAS

        # Background inference thread
        self._lock = threading.Lock()
        self._pending_frame: np.ndarray | None = None
        self._pending_use_face: bool = False
        self._latest_detections: list[tuple] = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---- background inference ------------------------------------------ #

    def _worker(self):
        """Continuously pull the latest pending frame and run YOLO on it."""
        while True:
            with self._lock:
                frame = self._pending_frame
                use_face = self._pending_use_face
                if frame is not None:
                    self._pending_frame = None

            if frame is None:
                time.sleep(0.002)
                continue

            model = self._model_face if use_face else self._model_person
            with _mute_stderr():
                results = model(
                    frame,
                    conf=config.YOLO_CONF_THRESHOLD,
                    classes=[0],
                    verbose=False,
                )

            detections: list[tuple] = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    area_index = self._area_for_cx(cx)
                    detections.append((cx, cy, area_index, (x1, y1, x2, y2)))

            with self._lock:
                self._latest_detections = detections

    # ---- main-thread update -------------------------------------------- #

    def _area_for_cx(self, cx: float) -> int:
        for boundary_idx in range(1, config.NUM_AREAS):
            bx = self._area_w * boundary_idx
            if cx < bx - self._dead_zone_px:
                return boundary_idx - 1
        return config.NUM_AREAS - 1

    def update(self, bgr_frame: np.ndarray, use_face_model: bool = False):
        self._frame_number += 1
        small = cv2.resize(bgr_frame, (self._frame_w_scaled, self._frame_h_scaled))

        # Submit frame to worker (non-blocking — drops previous pending if busy)
        with self._lock:
            self._pending_frame = small
            self._pending_use_face = use_face_model
            detections = list(self._latest_detections)

        # Nearest-neighbour tracking with latest detections
        max_dist = self._frame_w_scaled / 8
        matched_blob_ids: set[int] = set()
        new_blobs: list[_TrackedBlob] = []
        for cx, cy, area_idx, bbox in detections:
            best: _TrackedBlob | None = None
            best_d = max_dist
            for blob in self._blobs:
                if blob.blob_id in matched_blob_ids:
                    continue
                d = ((blob.cx - cx) ** 2 + (blob.cy - cy) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = blob
            if best is not None:
                best.cx, best.cy = cx, cy
                best.area_index = area_idx
                best.bbox = bbox
                best.y_history.append(cy)
                best.last_seen_frame = self._frame_number
                matched_blob_ids.add(best.blob_id)
                new_blobs.append(best)
                self._detect_jump(best)
            else:
                b = _TrackedBlob(cx, cy, area_idx)
                b.bbox = bbox
                b.last_seen_frame = self._frame_number
                new_blobs.append(b)

        # Retain recently seen blobs (dropout tolerance: 10 frames)
        for blob in self._blobs:
            if blob.blob_id not in matched_blob_ids:
                if self._frame_number - blob.last_seen_frame <= 10:
                    new_blobs.append(blob)

        self._blobs = new_blobs

    def _detect_jump(self, blob: _TrackedBlob):
        if len(blob.y_history) < 6:
            return
        history = list(blob.y_history)
        dy = history[-1] - history[-5]  # positive = moving down (y increases downward)
        now = time.monotonic()

        if blob._jump_state == "GROUND":
            if dy < -(self._frame_h_scaled * config.JUMP_Y_THRESHOLD):
                blob._jump_state = "RISING"
                blob._peak_y = blob.cy

        elif blob._jump_state == "RISING":
            if blob.cy < blob._peak_y:
                blob._peak_y = blob.cy
            if dy > (self._frame_h_scaled * config.JUMP_Y_THRESHOLD * 0.5):
                blob._jump_state = "FALLING"

        elif blob._jump_state == "FALLING":
            if dy >= 0 or blob.cy > blob._peak_y + self._frame_h_scaled * config.JUMP_Y_THRESHOLD * 0.5:
                if now - blob._last_jump_time > config.JUMP_DEBOUNCE_SEC:
                    self._jump_counts[blob.area_index] += 1
                    blob._last_jump_time = now
                blob._jump_state = "GROUND"

    # ---- accessors ----------------------------------------------------- #

    def get_area_counts(self) -> list[int]:
        counts = [0] * config.NUM_AREAS
        for blob in self._blobs:
            if self._frame_number - blob.last_seen_frame <= 2:
                counts[blob.area_index] += 1
        return counts

    def get_jump_counts(self) -> list[int]:
        counts = self._jump_counts[:]
        self._jump_counts = [0] * config.NUM_AREAS
        return counts

    def get_face_positions(self) -> list[tuple[int, int]]:
        inv = 1.0 / self._scale
        return [
            (int(blob.cx * inv), int(blob.cy * inv))
            for blob in self._blobs
            if self._frame_number - blob.last_seen_frame <= 2
        ]

    def is_spinning(self) -> bool:
        return False  # spin detection removed (unused)

    def get_debug_overlay(self, bgr_frame: np.ndarray) -> np.ndarray:
        out = bgr_frame.copy()
        for i in range(1, config.NUM_AREAS):
            x = config.WIDTH * i // config.NUM_AREAS
            cv2.line(out, (x, 0), (x, config.HEIGHT), (255, 255, 0), 2)
        inv = 1.0 / self._scale
        for blob in self._blobs:
            if self._frame_number - blob.last_seen_frame <= 2:
                if blob.bbox is not None:
                    x1, y1, x2, y2 = blob.bbox
                    cv2.rectangle(out, (int(x1 * inv), int(y1 * inv)),
                                  (int(x2 * inv), int(y2 * inv)), (0, 255, 0), 2)
                else:
                    cv2.circle(out, (int(blob.cx * inv), int(blob.cy * inv)), 12, (0, 255, 0), -1)
                cv2.putText(out, f"A{blob.area_index}",
                            (int(blob.cx * inv) + 14, int(blob.cy * inv) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return out
