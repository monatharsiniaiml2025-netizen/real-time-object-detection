
import cv2
from ultralytics import YOLOWorld

print("Loading YOLO-World model...")
model = YOLOWorld("yolov8s-worldv2.pt")

objects = [
    "person", "mobile phone", "laptop", "tablet",
    "necklace", "chain", "gold chain",
    "earphones", "headphones", "earbuds",
    "wristwatch", "eyeglasses", "sunglasses",
    "handbag", "backpack", "wallet",
    "bottle", "water bottle", "cup",
    "book", "pen", "pencil",
    "keyboard", "computer mouse", "charger",
    "remote control", "chair", "table",
    "keys", "shoe", "ring", "bracelet"
]

model.set_classes(objects)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opening. Close other camera apps and try again.")
    exit()

print("Camera started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera frame")
        break

    results = model.predict(
        frame,
        conf=0.10,
        imgsz=416,
        verbose=False
    )

    output = results[0].plot()
    cv2.imshow("Open Vocabulary Object Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
