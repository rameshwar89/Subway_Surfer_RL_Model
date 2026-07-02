import time

from android.capture import ScreenCapture
from controller.actions import SubwayActions


class SubwayEnv:

    def __init__(self):

        self.capture = ScreenCapture()
        self.actions = SubwayActions()

    def reset(self):

        frame = self.capture.grab()

        return frame

    def step(self, action):

        self.actions.execute(action)

        time.sleep(0.08)

        frame = self.capture.grab()

        reward = 0

        done = False

        info = {}

        return frame, reward, done, info