from android.capture import ScreenCapture

cap = ScreenCapture()

frame = cap.grab()

print(frame.shape)