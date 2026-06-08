import cv2
import numpy as np

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.framesync import FrameSync
from cyndilib.video_frame import VideoFrameSync
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth

finder = Finder()
receiver = Receiver(
    color_format=RecvColorFormat.RGBX_RGBA,
    bandwidth=RecvBandwidth.highest,
)

frame_sync = FrameSync(receiver)
video_frame = VideoFrameSync()
frame_sync.set_video_frame(video_frame)

# NDIソースを探す
finder.open()
finder.wait_for_sources(5000)

sources = finder.get_source_names()

if not sources:
    raise RuntimeError("NDI source not found")

print("NDI sources:")
for index, source_name in enumerate(sources, start=1):
    print(f"{index}: {source_name}")

while True:
    selected = input("Select NDI source number: ").strip()

    try:
        selected_index = int(selected) - 1
    except ValueError:
        print("Please enter a number.")
        continue

    if 0 <= selected_index < len(sources):
        break

    print(f"Please enter a number between 1 and {len(sources)}.")

selected_source_name = sources[selected_index]
print(f"Connecting to: {selected_source_name}")

# 選択したソースに接続
receiver.set_source(finder.get_source(selected_source_name))

while True:
    frame_sync.capture_video()

    xres, yres = video_frame.get_resolution()
    if xres == 0 or yres == 0:
        continue

    frame = video_frame.get_array()
    if frame is None or frame.size == 0:
        continue

    frame = frame.reshape(yres, xres, 4)

    # RGBX/RGBA → OpenCV用BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    cv2.imshow("NDI", bgr)

    if cv2.waitKey(1) == 27:
        break

receiver.disconnect()
finder.close()
cv2.destroyAllWindows()
