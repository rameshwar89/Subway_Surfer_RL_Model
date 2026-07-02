from android.capture import ScreenCapture
from rl.observation import ObservationProcessor
import cv2

capture = ScreenCapture()
processor = ObservationProcessor()

frame = capture.grab()
obs = processor.process(frame)

print("Original:", frame.shape)
print("Processed:", obs.shape)

debug = cv2.resize(
    obs,
    (512, 512),
    interpolation=cv2.INTER_NEAREST,
)

cv2.imshow("Processed", debug)
cv2.waitKey(0)
cv2.destroyAllWindows()