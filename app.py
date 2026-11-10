
import streamlit as st
import cv2
import av
import torch
import threading
import queue

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase


st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="📷",
    layout="wide"
)

st.title("Real-Time Object Detection")
st.write("Detects only five selected objects using YOLOv8n.")

TARGET_CLASSES = {
    "cell phone",
    "bottle",
    "remote",
    "book",
    "scissors"
}


@st.cache_resource
def load_model():
    torch.set_num_threads(2)
    return YOLO("yolov8n.pt")


try:
    model = load_model()
    st.success("YOLOv8n model loaded successfully!")
except Exception as e:
    st.error(f"Model error: {e}")
    st.stop()


class ObjectDetectionProcessor(VideoProcessorBase):

    def __init__(self, model):
        self.model = model
        self.frame_count = 0
        self.frame_queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()

        self.boxes = []
        self.status = "Waiting for detection"
        self.error = ""

        self.worker = threading.Thread(
            target=self.detect_objects,
            daemon=True
        )
        self.worker.start()

    def detect_objects(self):
        while True:
            image = self.frame_queue.get()

            try:
                results = self.model.predict(
                    source=image,
                    conf=0.50,
                    iou=0.45,
                    imgsz=416,
                    max_det=10,
                    classes=[0,67, 39, 65, 73, 76],
                    device="cpu",
                    verbose=False
                )

                detected = []

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(
                            int, box.xyxy[0].tolist()
                        )

                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        label = result.names[class_id]

                        if label in TARGET_CLASSES:
                            detected.append(
                                (x1, y1, x2, y2, label, confidence)
                            )

                with self.lock:
                    self.boxes = detected
                    self.status = (
                        f"{len(detected)} selected objects detected"
                        if detected else "No selected object detected"
                    )
                    self.error = ""

            except Exception as e:
                with self.lock:
                    self.status = "Detection error"
                    self.error = str(e)

                print("Detection error:", e)

            finally:
                self.frame_queue.task_done()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Process every 5th frame
        if self.frame_count % 5 == 0:
            try:
                self.frame_queue.put_nowait(image.copy())
            except queue.Full:
                pass

        output = image.copy()

        with self.lock:
            boxes = list(self.boxes)
            status = self.status

        for x1, y1, x2, y2, label, confidence in boxes:
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = f"{label}: {confidence * 100:.1f}%"

            cv2.putText(
                output,
                text,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.rectangle(
            output,
            (0, 0),
            (output.shape[1], 40),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            output,
            status,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            output,
            format="bgr24"
        )


st.subheader("Live Webcam")

webrtc_streamer(
    key="object-detection-five-classes",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: ObjectDetectionProcessor(model),
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 20, "max": 24}
        },
        "audio": False
    },
    async_processing=True
)

st.divider()

st.subheader("Selected Objects")

st.write(
    "Cellphone | Bottle | Remote | Book | Scissors"
)

st.caption("Model: YOLOv8n | Confidence: 50% | Device: CPU")
st.info(
    "Only the five selected classes are displayed. "
    "Detection accuracy depends on object visibility, lighting and camera quality."
)
