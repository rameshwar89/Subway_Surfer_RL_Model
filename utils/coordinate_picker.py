import cv2

from android.capture import ScreenCapture

capture = ScreenCapture()

window = "Coordinate Picker"


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        frame = param["frame"]

        h, w = frame.shape[:2]

        # Convert display coordinates back to original resolution
        ox = int(x * w / param["display_w"])
        oy = int(y * h / param["display_h"])

        print(f"\nClicked:")
        print(f"Display : ({x}, {y})")
        print(f"Original: ({ox}, {oy})")

        cv2.circle(param["display"], (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            param["display"],
            f"({ox},{oy})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )


state = {
    "frame": None,
    "display": None,
    "display_w": 400,
    "display_h": 900,
}

cv2.namedWindow(window)
cv2.setMouseCallback(window, mouse_callback, state)

while True:

    frame = capture.grab()

    state["frame"] = frame

    display = cv2.resize(
        frame,
        (state["display_w"], state["display_h"])
    )

    state["display"] = display

    cv2.imshow(window, display)

    key = cv2.waitKey(30)

    if key == 27:
        break

cv2.destroyAllWindows()