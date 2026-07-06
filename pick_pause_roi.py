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

    print("\n======================================")
    print("Drag a rectangle around the PAUSE BUTTON.")
    print("Press ENTER when finished.")
    print("Press C to cancel.")
    print("======================================")

    x, y, w, h = cv2.selectROI(
        "Select Pause Button ROI",
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

    # Load calibration.json
    with open("configs/calibration.json", "r") as f:
        config = json.load(f)

    # Update pause_button ROI
    config["state_detection"]["rois"]["pause_button"] = {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
    }

    # Save calibration.json
    with open("configs/calibration.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nSaved Pause Button ROI to calibration.json:\n")
    print(json.dumps(config["state_detection"]["rois"]["pause_button"], indent=4))

    cropped = frame[y:y+h, x:x+w]
    cv2.imshow("Selected Pause Button ROI", cropped)
    
    # Save the patch directly as well so it's perfectly accurate!
    cv2.imwrite("assets/patches/ui/pause_button/pause_button_active.png", cropped)
    print("Saved new patch image to assets/patches/ui/pause_button/pause_button_active.png!")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
