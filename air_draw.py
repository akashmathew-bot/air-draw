import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape

# Canvas to draw on
canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_x, prev_y = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # Index fingertip = landmark 8, index base = landmark 5 (to check if finger is up)
        tip = landmarks[8]
        base = landmarks[5]

        x, y = int(tip.x * w), int(tip.y * h)

        # Only draw if index finger is extended upward (tip above base)
        if tip.y < base.y:
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None

        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
    else:
        prev_x, prev_y = None, None

    # Combine canvas with frame
    combined = cv2.addWeighted(frame, 0.7, canvas, 1, 0)

    cv2.putText(combined, "Press 'c' to clear, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Air Drawing", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()