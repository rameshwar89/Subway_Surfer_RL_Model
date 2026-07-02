import time

from android.adb_controller import ADBController

adb = ADBController()

print("You have 5 seconds...")
time.sleep(5)

adb.shell(
    "input",
    "tap",
    "845",
    "2298",
)

print("Tapped.")