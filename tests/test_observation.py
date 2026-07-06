from android.capture import ScreenCapture
from rl.observation import ObservationProcessor

import cv2
import numpy as np

capture = ScreenCapture()
processor = ObservationProcessor()

frame = capture.grab()
obs = processor.process(frame)

print("========== Observation Test ==========")
print("Original Shape :", frame.shape)
print("Processed Shape:", obs.shape)
print("Processed Dtype:", obs.dtype)
print("Min Pixel      :", obs.min())
print("Max Pixel      :", obs.max())

assert obs.shape == (
    processor.height,
    processor.width,
    1,
)

assert obs.dtype == np.uint8

assert 0 <= obs.min() <= 255
assert 0 <= obs.max() <= 255

debug = (obs.squeeze() * 255).astype(np.uint8)

debug = cv2.resize(
    debug,
    (512, 512),
    interpolation=cv2.INTER_NEAREST,
)

cv2.imshow("Processed Observation", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✅ Observation test passed!")