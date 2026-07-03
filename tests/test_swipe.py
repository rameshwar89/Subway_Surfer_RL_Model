import time
import json

from android.transport import AndroidTransport

transport = AndroidTransport()

with open("configs/calibration.json") as f:
    cfg = json.load(f)

print("Waiting 3 seconds...")

time.sleep(3)

print("LEFT")

transport.swipe(
    cfg["center_x"],
    cfg["center_y"],
    cfg["left_x"],
    cfg["center_y"],
)

time.sleep(2)

print("RIGHT")

transport.swipe(
    cfg["center_x"],
    cfg["center_y"],
    cfg["right_x"],
    cfg["center_y"],
)

time.sleep(2)

transport.stop()