from collections import deque
import time

import cv2
import numpy as np

import config


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


class MotionDetector:
    def __init__(self):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )
        self._scale = config.DETECTION_SCALE
        self._area_w = int(config.WIDTH * self._scale / config.NUM_AREAS)
        self._frame_h_scaled = int(config.HEIGHT * self._scale)
        self._frame_w_scaled = int(config.WIDTH * self._scale)
        self._dead_zone_px = int(config.WIDTH * self._scale * config.AREA_DEAD_ZONE)

        self._blobs: list[_TrackedBlob] = []
        self._frame_number = 0
        self._jump_counts = [0] * config.NUM_AREAS

        # Spin detection state
        self._prev_gray: np.ndarray | None = None
        self._spin_consec_frames = 0
        self._spin_frames_needed = int(config.SPIN_DURATION_SEC * 30)

    def _area_for_cx(self, cx: float) -> int:
        for boundary_idx in range(1, config.NUM_AREAS):
            bx = self._area_w * boundary_idx
            if cx < bx - self._dead_zone_px:
                return boundary_idx - 1
        return config.NUM_AREAS - 1

    def update(self, bgr_frame: np.ndarray):
        self._frame_number += 1

        small = cv2.resize(bgr_frame, (self._frame_w_scaled, self._frame_h_scaled))
        fg_mask = self._bg.apply(small)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[tuple[float, float, int]] = []  # (cx, cy, area_index)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_PERSON_BLOB_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            area_index = self._area_for_cx(cx)
            detections.append((cx, cy, area_index))

        # Nearest-neighbour tracking
        max_dist = self._frame_w_scaled / 8
        matched_blob_ids = set()
        new_blobs: list[_TrackedBlob] = []
        for cx, cy, area_idx in detections:
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
                best.y_history.append(cy)
                best.last_seen_frame = self._frame_number
                matched_blob_ids.add(best.blob_id)
                new_blobs.append(best)
                self._detect_jump(best)
            else:
                b = _TrackedBlob(cx, cy, area_idx)
                b.last_seen_frame = self._frame_number
                new_blobs.append(b)

        # Keep old blobs that disappeared recently (up to 10 frames dropout tolerance)
        for blob in self._blobs:
            if blob.blob_id not in matched_blob_ids:
                if self._frame_number - blob.last_seen_frame <= 10:
                    new_blobs.append(blob)

        self._blobs = new_blobs

        # Spin detection via optical flow
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            mean_flow = float(np.mean(np.abs(flow[..., 0])))
            if mean_flow > config.SPIN_FLOW_THRESHOLD:
                self._spin_consec_frames += 1
            else:
                self._spin_consec_frames = max(0, self._spin_consec_frames - 1)
        self._prev_gray = gray

    def _detect_jump(self, blob: _TrackedBlob):
        if len(blob.y_history) < 5:
            return
        history = list(blob.y_history)
        dy = history[-1] - history[-3]  # positive = moving down in image (y increases downward)
        now = time.monotonic()

        if blob._jump_state == "GROUND":
            # dy negative means person moved up (Y decreased)
            if dy < -(self._frame_h_scaled * config.JUMP_Y_THRESHOLD):
                blob._jump_state = "RISING"
                blob._peak_y = blob.cy

        elif blob._jump_state == "RISING":
            if blob.cy < blob._peak_y:
                blob._peak_y = blob.cy
            # dy positive = coming back down
            if dy > (self._frame_h_scaled * config.JUMP_Y_THRESHOLD * 0.5):
                blob._jump_state = "FALLING"

        elif blob._jump_state == "FALLING":
            if dy >= 0 or blob.cy > blob._peak_y + self._frame_h_scaled * config.JUMP_Y_THRESHOLD * 0.5:
                if now - blob._last_jump_time > config.JUMP_DEBOUNCE_SEC:
                    self._jump_counts[blob.area_index] += 1
                    blob._last_jump_time = now
                blob._jump_state = "GROUND"

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

    def is_spinning(self) -> bool:
        return self._spin_consec_frames >= self._spin_frames_needed

    def get_debug_overlay(self, bgr_frame: np.ndarray) -> np.ndarray:
        out = bgr_frame.copy()
        # Area dividers
        for i in range(1, config.NUM_AREAS):
            x = config.WIDTH * i // config.NUM_AREAS
            cv2.line(out, (x, 0), (x, config.HEIGHT), (255, 255, 0), 2)
        # Blob centroids (scale back up)
        inv = 1.0 / self._scale
        for blob in self._blobs:
            if self._frame_number - blob.last_seen_frame <= 2:
                cx = int(blob.cx * inv)
                cy = int(blob.cy * inv)
                cv2.circle(out, (cx, cy), 12, (0, 255, 0), -1)
                cv2.putText(out, f"A{blob.area_index}", (cx + 14, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return out
