
import cv2
from ultralytics import YOLO
import time

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Webcam not opening!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Webcam started. Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read webcam frame")
        break

    results = model.predict(
        frame,
        conf=0.15,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    output = result.plot()

    detected = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]
            detected.append(name)

    cv2.putText(
        output,
        f"Objects: {len(detected)}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Webcam Test", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Webcam test completed.")
