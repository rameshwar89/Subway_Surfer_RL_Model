import json
import cv2

from android.capture import ScreenCapture


def main():

    capture = ScreenCapture()

    print("Capturing frame...")

    frame = capture.grab()

    if frame is None:
        print("Failed to capture frame.")
        return

    # Resize preview if the screen is too large
    preview_height = 900
    scale = preview_height / frame.shape[0]

    preview = cv2.resize(
        frame,
        (
            int(frame.shape[1] * scale),
            preview_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    print("\n")
    print("======================================")
    print("Drag a rectangle around ONLY")
    print("the gameplay area.")
    print("Press ENTER when finished.")
    print("Press C to cancel.")
    print("======================================")

    x, y, w, h = cv2.selectROI(
        "Select Gameplay ROI",
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("ROI cancelled.")
        return

    # Convert back to original resolution
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    config = {
        "crop_x": x,
        "crop_y": y,
        "crop_width": w,
        "crop_height": h,
    }

    with open("assets/configs/observation.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nSaved Observation ROI:\n")
    print(json.dumps(config, indent=4))

    cropped = frame[y:y+h, x:x+w]

    cv2.imshow("Selected Gameplay ROI", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()