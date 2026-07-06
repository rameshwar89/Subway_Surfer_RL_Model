from controller.actions import SubwayActions
import time

a = SubwayActions()

print("LEFT")
a.execute(0)
time.sleep(2)

print("RIGHT")
a.execute(1)
time.sleep(2)

print("JUMP")
a.execute(2)
time.sleep(2)

print("ROLL")
a.execute(3)