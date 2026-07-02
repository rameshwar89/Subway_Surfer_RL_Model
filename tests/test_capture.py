import cv2

from android.capture import ScreenCapture


cap = ScreenCapture()

frame = cap.grab()

cv2.imshow("Screen", frame)

cv2.waitKey(0)