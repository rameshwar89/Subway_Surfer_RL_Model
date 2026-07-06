import json
from pathlib import Path

import cv2

from android.capture import ScreenCapture


DISPLAY_WIDTH = 360
DISPLAY_HEIGHT = 800
DEFAULT_SAVE_DIR = Path("assets/patches")

CATEGORIES = {
    "1": DEFAULT_SAVE_DIR / "popup" / "use_keys",
    "2": DEFAULT_SAVE_DIR / "ui" / "game_over",
    "3": DEFAULT_SAVE_DIR / "ui" / "main_menu",
    "4": DEFAULT_SAVE_DIR / "ui" / "pause_menu",
    "5": DEFAULT_SAVE_DIR / "ui" / "leave_confirm",
}


capture = ScreenCapture()

drawing = False
start_point = None
end_point = None
frame = None
display_frame = None


def capture_frame():

    grabbed = capture.grab()

    return grabbed.copy()


def reset_selection():

    global drawing, start_point, end_point

    drawing = False
    start_point = None
    end_point = None


def mouse_callback(event, x, y, flags, param):

    global drawing, start_point, end_point

    if frame is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)


def frame_to_display(image):

    return cv2.resize(
        image,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
    )


def display_to_frame_rect():

    h, w = frame.shape[:2]

    scale_x = w / DISPLAY_WIDTH
    scale_y = h / DISPLAY_HEIGHT

    x1 = int(min(start_point[0], end_point[0]) * scale_x)
    y1 = int(min(start_point[1], end_point[1]) * scale_y)
    x2 = int(max(start_point[0], end_point[0]) * scale_x)
    y2 = int(max(start_point[1], end_point[1]) * scale_y)

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    return x1, y1, x2, y2


def choose_save_dir():

    print("\nPatch category:")
    print("  1 popup/use_keys")
    print("  2 ui/game_over")
    print("  3 ui/main_menu")
    print("  4 ui/pause_menu")
    print("  5 ui/leave_confirm")

    choice = input("Choose category [1]: ").strip() or "1"

    return CATEGORIES.get(
        choice,
        DEFAULT_SAVE_DIR / "popup" / "use_keys",
    )


def save_crop():

    if frame is None:
        print("No captured frame. Press C first.")
        return

    if start_point is None or end_point is None:
        print("No ROI selected.")
        return

    x1, y1, x2, y2 = display_to_frame_rect()

    if x2 <= x1 or y2 <= y1:
        print("Invalid ROI.")
        return

    roi = frame[y1:y2, x1:x2]

    save_dir = choose_save_dir()
    save_dir.mkdir(parents=True, exist_ok=True)

    name = input("Enter patch filename without extension: ").strip()

    if name == "":
        print("Invalid filename.")
        return

    image_path = save_dir / f"{name}.png"
    json_path = save_dir / f"{name}.json"

    cv2.imwrite(str(image_path), roi)

    metadata = {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }

    with json_path.open("w") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved crop")
    print(image_path)
    print(json_path)
    print(metadata)


def render():

    if frame is None:
        preview = capture.grab()
        view = frame_to_display(preview)
        text = "Live preview | C=capture | Q=quit"
    else:
        view = display_frame.copy()
        text = "Frozen frame | Drag crop | S=save | C=recapture | R=reset | Q=quit"

    if start_point and end_point:
        cv2.rectangle(
            view,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        view,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
    )

    cv2.imshow("ROI Picker", view)


cv2.namedWindow("ROI Picker")
cv2.setMouseCallback("ROI Picker", mouse_callback)

print("=" * 60)
print("ROI PICKER")
print("=" * 60)
print("Workflow:")
print("  1. Press C to capture/freeze the current frame")
print("  2. Drag the crop box on the frozen frame")
print("  3. Press S to save the cropped patch")
print("Controls:")
print("  C  Capture / recapture still frame")
print("  S  Save selected crop")
print("  R  Reset selection")
print("  Q  Quit")
print("=" * 60)

while True:

    render()

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):

        frame = capture_frame()
        display_frame = frame_to_display(frame)
        reset_selection()
        print("Captured still frame.")

    elif key == ord("r"):

        reset_selection()

    elif key == ord("s"):

        save_crop()
        reset_selection()

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
