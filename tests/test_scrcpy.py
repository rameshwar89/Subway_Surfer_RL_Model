import time
import cv2
import scrcpy

latest_frame = None

def on_frame(frame):
    global latest_frame

    print(type(frame))

    if frame is None:
        print("Frame is None")
        return

    print("Shape:", frame.shape)

    latest_frame = frame


client = scrcpy.Client(
    max_width=720,
    bitrate=8000000,
    max_fps=30,
)

client.add_listener(scrcpy.EVENT_FRAME, on_frame)

client.start(threaded=True)

print("Waiting for stream...")

while latest_frame is None:
    time.sleep(0.1)

print("Stream started!")
while True:

    if latest_frame is None:
        time.sleep(0.05)
        continue

    print(latest_frame.shape)

    cv2.imshow("Scrcpy Stream", latest_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()