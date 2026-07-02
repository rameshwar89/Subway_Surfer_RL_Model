import cv2
import time

from rl.subway_env import SubwayEnv


env = SubwayEnv()

obs = env.reset()

sequence = [2, 0, 1, 3]

for action in sequence:

    obs, reward, done, info = env.step(action)

    cv2.imshow("Observation", obs)

    cv2.waitKey(1)

    time.sleep(1)

cv2.destroyAllWindows()