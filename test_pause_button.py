import cv2
import time
from vision.state_detector import StateDetector
from android.capture import ScreenCapture

def test_detector():
    print("Initializing Capture and Detector...")
    capture = ScreenCapture()
    detector = StateDetector()
    
    # Get the ROI so we can draw it
    roi = detector.pause_button_detector.roi
    x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
    
    cv2.namedWindow("Pause Detector Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pause Detector Test", 400, 800)
    
    print("Test window opened! Please start the game and press 'q' to quit.")
    
    # Run indefinitely until user presses 'q'
    frame_count = 0
    while True:
        frame = capture.grab()
        t0 = time.perf_counter()
        
        state, votes, scores = detector.detect(frame, context="gameplay")
        
        t1 = time.perf_counter()
        
        pause_debug = detector.last_debug.get("pause_button", {})
        matched = pause_debug.get("matched", False)
        score = pause_debug.get("score", 0.0)
        
        # Make a copy for drawing
        display = frame.copy()
        
        # Draw ROI
        color = (0, 255, 0) if matched else (0, 0, 255)
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 4)
        
        # Overlay text
        status_text = "MATCH" if matched else "MISSING"
        cv2.putText(display, f"Pause Button: {status_text}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(display, f"Score: {score:.3f} (Req: 0.68)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Frame {frame_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow("Pause Detector Test", display)
        
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        time.sleep(0.05)
        
    cv2.destroyAllWindows()
        
if __name__ == "__main__":
    test_detector()
