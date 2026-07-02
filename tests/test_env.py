import cv2
import time

from rl.subway_env import SubwayEnv

env = SubwayEnv()

obs = env.reset()

actions = [

    2,   # jump

    0,   # left

    1,   # right

    3,   # roll

    4,   # idle
]

for action in actions:

    obs, reward, done, info = env.step(action)

    print(
        reward,
        done,
        info,
    )

    display = cv2.resize(obs, (360, 800))

    cv2.imshow(
        "Observation",
        display,
    )

    cv2.waitKey(1)

    time.sleep(1)

cv2.destroyAllWindows()