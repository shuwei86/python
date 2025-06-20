import cv2
import numpy as np
from datetime import datetime


def detect_fire_smoke(frame):
    """Detect flame or smoke in the frame using naive color thresholds.

    Returns a tuple (has_fire, has_smoke).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Fire detection (red, orange, yellow hues)
    lower_fire1 = np.array([0, 100, 100])
    upper_fire1 = np.array([10, 255, 255])
    lower_fire2 = np.array([160, 100, 100])
    upper_fire2 = np.array([179, 255, 255])
    fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
    fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)

    # Additional range for yellow/orange
    lower_fire3 = np.array([11, 100, 100])
    upper_fire3 = np.array([35, 255, 255])
    fire_mask3 = cv2.inRange(hsv, lower_fire3, upper_fire3)
    fire_mask = cv2.bitwise_or(fire_mask, fire_mask3)

    fire_pixels = cv2.countNonZero(fire_mask)
    has_fire = fire_pixels > 500  # threshold of pixel count

    # Smoke detection: look for low saturation but high value (grey/white areas)
    lower_smoke = np.array([0, 0, 150])
    upper_smoke = np.array([179, 50, 255])
    smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
    smoke_pixels = cv2.countNonZero(smoke_mask)
    has_smoke = smoke_pixels > 1000  # threshold of pixel count

    return has_fire, has_smoke


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    log = open("detection_log.txt", "a")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            has_fire, has_smoke = detect_fire_smoke(frame)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = []
            if has_fire:
                message.append("FIRE")
            if has_smoke:
                message.append("SMOKE")

            if message:
                label = f"{', '.join(message)} {timestamp}"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                log.write(label + "\n")
                log.flush()

            cv2.imshow("Fire & Smoke Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
