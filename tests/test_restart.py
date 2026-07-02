import time

from controller.game_controller import GameController

gc = GameController()

print("Die first...")
time.sleep(5)

gc.restart()

print("Done.")