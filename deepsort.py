from typing import Any
from ultralytics import YOLO
import cv2
import cvzone
import supervision as sv
import pandas as pd
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import math


class ObjectDetection:
    def __init__(self, folder, width, height):
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.width = width
        self.height = height
        self.folder = folder

    def load_model(self):
        model = YOLO("yolov8n.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img)
        return results

    def plot_boxes(self, results, img):
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                # classname
                cls = int(box.cls[0])
                current_class = self.CLASS_NAMES_DICT[cls]

                # confidence score
                conf = math.ceil(box.conf[0] * 100) / 100

                if (
                    current_class == "person"
                    and y1 > self.height * 0.2
                    and y1 < self.height * 0.8
                ):
                    detections.append((([x1, y1, w, h]), conf, current_class))

        return detections, img

    def track_detect(self, detections, img, tracker):
        tracks = tracker.update_tracks(detections, frame=img)

        ids = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ids.append(track_id)
            ltrb = track.to_ltrb()

            bbox = ltrb
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cvzone.putTextRect(
                img,
                f"#: {track_id} - person",
                (x1, y1),
                scale=1,
                thickness=1,
                colorR=(0, 0, 255),
            )
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))

        return img, ids

    def save_results(self):
        pass

    def __call__(self):
        files = sorted(os.listdir("data/pending/" + self.folder))
        tracker = DeepSort(
            max_age=2,
            n_init=2,
            nms_max_overlap=1,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None,
        )
        for file in files:
            img = cv2.imread("data/pending/" + self.folder + "/" + file)
            img_resized = cv2.resize(img, (self.width, self.height))
            # cv2.imshow("Image", img_resized)
            # cv2.waitKey(0)

            img_rotated = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)

            # cv2.imshow("Image", img_rotated)
            # cv2.waitKey(0)

            results = self.predict(img_rotated)

            detections, frames = self.plot_boxes(results, img_rotated)
            detect_frame, ids = self.track_detect(detections, frames, tracker)

            cv2.imshow("Image", detect_frame)

            cv2.waitKey(0)

        cv2.destroyAllWindows()


def main():
    detector = ObjectDetection("2023-10-12", 480, 720)
    detector()


if __name__ == "__main__":
    main()
