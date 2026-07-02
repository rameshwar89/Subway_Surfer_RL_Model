import json
import time

from android.gestures import Gestures


with open("configs/calibration.json") as f:
    cfg = json.load(f)

g = Gestures()

time.sleep(2)

print("LEFT")
g.swipe(
    cfg["center_x"],
    cfg["center_y"],
    cfg["left_x"],
    cfg["center_y"],
    cfg["swipe_duration"],
)

time.sleep(1)

print("RIGHT")
g.swipe(
    cfg["center_x"],
    cfg["center_y"],
    cfg["right_x"],
    cfg["center_y"],
    cfg["swipe_duration"],
)

time.sleep(1)

print("JUMP")
g.swipe(
    cfg["center_x"],
    cfg["center_y"],
    cfg["center_x"],
    cfg["jump_y"],
    cfg["swipe_duration"],
)

time.sleep(1)

print("ROLL")
g.swipe(
    cfg["center_x"],
    cfg["center_y"],
    cfg["center_x"],
    cfg["roll_y"],
    cfg["swipe_duration"],
)