import os
import json
import cv2

from android.capture import ScreenCapture


DISPLAY_WIDTH = 360
DISPLAY_HEIGHT = 800

SAVE_DIR = "vision/references"
os.makedirs(SAVE_DIR, exist_ok=True)

capture = ScreenCapture()

drawing = False
start_point = None
end_point = None
frame = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)


cv2.namedWindow("ROI Picker")
cv2.setMouseCallback("ROI Picker", mouse_callback)

print("=" * 60)
print("ROI PICKER")
print("=" * 60)
print("Controls:")
print("  Drag Left Mouse  -> Select ROI")
print("  S                -> Save ROI")
print("  R                -> Reset Selection")
print("  Q                -> Quit")
print("=" * 60)

while True:

    frame = capture.grab()

    display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    if start_point and end_point:
        cv2.rectangle(
            display,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        display,
        "Drag ROI | S=Save | R=Reset | Q=Quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.imshow("ROI Picker", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):

        start_point = None
        end_point = None

    elif key == ord("s"):

        if start_point is None or end_point is None:
            print("No ROI selected.")
            continue

        H, W = frame.shape[:2]

        scale_x = W / DISPLAY_WIDTH
        scale_y = H / DISPLAY_HEIGHT

        x1 = int(min(start_point[0], end_point[0]) * scale_x)
        y1 = int(min(start_point[1], end_point[1]) * scale_y)

        x2 = int(max(start_point[0], end_point[0]) * scale_x)
        y2 = int(max(start_point[1], end_point[1]) * scale_y)

        roi = frame[y1:y2, x1:x2]

        name = input("\nEnter filename: ").strip()

        if name == "":
            print("Invalid filename.")
            continue

        image_path = os.path.join(SAVE_DIR, f"{name}.png")
        json_path = os.path.join(SAVE_DIR, f"{name}.json")

        cv2.imwrite(image_path, roi)

        metadata = {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1
        }

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print("\nSaved Successfully")
        print(image_path)
        print(json_path)
        print(metadata)

        start_point = None
        end_point = None

    elif key == ord("q"):
        break

cv2.destroyAllWindows()