import multiprocessing

import numpy as np
import cv2

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.framesync import FrameSync
from cyndilib.video_frame import VideoFrameSync
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth


def _find_ndi_sources_worker(result_queue: multiprocessing.Queue, wait_ms: int) -> None:
    """Subprocess worker: discover NDI sources and put names in result_queue.

    Runs in a separate process to avoid GIL deadlock caused by cyndilib's
    wait_for_sources blocking indefinitely in its C extension.
    """
    try:
        from cyndilib.finder import Finder
        f = Finder()
        f.open()
        f.wait_for_sources(wait_ms)
        names = list(f.get_source_names())
        f.close()
        result_queue.put(names)
    except Exception:
        result_queue.put([])


class NDIReceiver:
    def __init__(self):
        self._receiver = Receiver(
            color_format=RecvColorFormat.RGBX_RGBA,
            bandwidth=RecvBandwidth.highest,
        )
        self._finder = Finder()
        self._frame_sync = FrameSync(self._receiver)
        self._video_frame = VideoFrameSync()
        self._frame_sync.set_video_frame(self._video_frame)
        self._connected = False
        self._finder_opened = False

    def list_sources(self, wait_ms: int = 5000) -> list[str]:
        # Also open main finder so it can discover sources concurrently
        if not self._finder_opened:
            self._finder.open()
            self._finder_opened = True

        # Use a subprocess so the process-level join respects the timeout
        # (threading won't work because wait_for_sources holds the GIL)
        ctx = multiprocessing.get_context("spawn")
        q: multiprocessing.Queue = ctx.Queue()
        p = ctx.Process(target=_find_ndi_sources_worker, args=(q, wait_ms), daemon=True)
        p.start()
        p.join(timeout=wait_ms / 1000 + 2.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)
            return []
        try:
            return q.get_nowait()
        except Exception:
            return []

    def connect(self, source_name: str):
        source = self._finder.get_source(source_name)
        self._receiver.set_source(source)
        self._connected = True

    def connect_by_index(self, index: int, wait_ms: int = 5000) -> str:
        sources = self.list_sources(wait_ms)
        if not sources:
            raise RuntimeError("NDIソースが見つかりません")
        if index >= len(sources):
            raise IndexError(f"インデックス {index} は範囲外です (利用可能: {len(sources)})")
        self.connect(sources[index])
        return sources[index]

    def connect_interactive(self, wait_ms: int = 5000) -> str:
        sources = self.list_sources(wait_ms)
        if not sources:
            raise RuntimeError("NDIソースが見つかりません")
        print("NDIソース一覧:")
        for i, name in enumerate(sources, start=1):
            print(f"  {i}: {name}")
        while True:
            raw = input("番号を入力してください: ").strip()
            try:
                idx = int(raw) - 1
            except ValueError:
                print("数字を入力してください")
                continue
            if 0 <= idx < len(sources):
                self.connect(sources[idx])
                return sources[idx]
            print(f"1〜{len(sources)} の番号を入力してください")

    def get_frame(self) -> np.ndarray | None:
        self._frame_sync.capture_video()
        xres, yres = self._video_frame.get_resolution()
        if xres == 0 or yres == 0:
            return None
        arr = self._video_frame.get_array()
        if arr is None or arr.size == 0:
            return None
        rgba = arr.reshape(yres, xres, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def close(self):
        if self._connected:
            self._receiver.disconnect()
        if self._finder_opened:
            self._finder.close()
