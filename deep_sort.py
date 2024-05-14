from ultralytics import YOLO
import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker
from stores_config import stores
import argparse
import os
import sys

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'

def main(folder, store, outputFolder):
   # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    tracker = DeepSort(model_path=deep_sort_weights, max_age=5)

    frames = []
    unique_track_ids = set()

    i = 0
    counter, fps, elapsed = 0, 0, 0
    start_time = time.perf_counter()

    info = []

    # init dataframe data
    hours = []
    ids = []
    agenders = []

    height = 480
    width = 720

    store_set = stores[store]
    print("store: ", store_set)

    images = sorted(os.listdir(args.imagesFolder))
    # print("images: ", images)

    for image in images:
        img = cv2.imread(folder + image, cv2.IMREAD_COLOR)

        img_rotated = cv2.resize(img, (height, width))
        color = (0, 255, 0)


        if store_set["rotate"] is not None:
            img_rotated = cv2.rotate(img_rotated, store_set["rotate"])

        if store_set["crop"] is not None:
            img_rotated = img_rotated[store_set["crop"]["height"][0]:store_set["crop"]["height"][1] ,
                                      store_set["crop"]["width"][0]:store_set["crop"]["width"][1]]

        start_point = (store_set["polygon"][0][0], store_set["polygon"][0][1])
        end_point = (store_set["polygon"][2][0], store_set["polygon"][2][1])
        img_rotated = cv2.rectangle(img_rotated, start_point, end_point, color, thickness=2)
        
        # detect + track
        results = model.predict(img_rotated, conf=0.6, classes=0)

        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            for class_index in cls:
                class_name = class_names[int(class_index)]
                #print("Class:", class_name)

       

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        print("conf: ", conf)
        print("conf type: ", type(conf))
        print("conf shape: ", conf.shape)


        xyxy = xyxy.detach().cpu().numpy()

        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)
        
        bboxes_results = []
        conf_results = []
        print("bboxes_xywh: ", bboxes_xywh)
        print("bbxywh type: ", type(bboxes_xywh))
        print("bbxywh shape: ", bboxes_xywh.shape)
        for index, bbox_xywh in enumerate(bboxes_xywh):
            # upper_centroid = (bbox_xywh[0]+(bbox_xywh[2]/2), bbox_xywh[1]+(bbox_xywh[3]/2))
            upper_centroid = (bbox_xywh[2]/2, bbox_xywh[1])

            if (upper_centroid[1] > store_set["coords"]["y1"]) and (upper_centroid[1] < store_set["coords"]["y2"]):
                bboxes_results.append(bbox_xywh)
                conf_results.append(conf[index])

        if len(bboxes_results) == 0:
            bboxes_xywh2 = np.ndarray(shape=(0, 4))
        else:
            bboxes_xywh2 = np.array(bboxes_results)

        conf2 = np.array(conf_results)

        print("#"*40)
        print("bboxes_xywh2: ", bboxes_xywh2)
        print("bbxywh2 type: ", type(bboxes_xywh2))
        print("bbxywh2 shape: ", bboxes_xywh2.shape)

        tracks = tracker.update(bboxes_xywh2, conf2, img_rotated)
        
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # Set color values for red, blue, and green
            red_color = (0, 0, 255)  # (B, G, R)
            blue_color = (255, 0, 0)  # (B, G, R)
            green_color = (0, 255, 0)  # (B, G, R)

            # Determine color based on track_id
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
            elif color_id == 1:
                color = blue_color
            else:
                color = green_color

            cv2.rectangle(img_rotated, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(img_rotated, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            # Add the track_id to the set of unique track IDs
            unique_track_ids.add(track_id)

        # Update the person count based on the number of unique track IDs
        person_count = len(unique_track_ids)

        # Update FPS and place on frame
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

        # Draw person count on frame
        cv2.putText(img_rotated, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Append the frame to the list
        # frames.append(og_frame)

        # if(detections.__len__() > 0):
        #     agender = DeepFace.analyze(img_rotated, actions=("gender", "age"),  detector_backend="ssd", enforce_detection=False)
        #     print("agender: ", agender)

        #     img_rotated, info = compare_centroids(detections, agender, store_set["coords"], img_rotated)
            
        print("#"*100)

        
        # # add to results
        # splitted_image = image.split(".")
        # splitted_folder = folder.split("/")
        # print("splitted_folder: ", splitted_folder)
        # format_date = splitted_folder[-2] + " " + splitted_image[0].replace("-", ":")
        # date = pd.to_datetime(format_date, format="%Y-%m-%d %H:%M:%S")
        # hours.append(date)
        # ids.append(detections.tracker_id)
        # agenders.append(info)
        
        # cv2.imwrite(outputFolder + "results/"+ image, img_rotated)

        cv2.imshow("yolov8", img_rotated)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            sys.exit()

        
    # print("Generating dataframe...")
    # df = pd.DataFrame(columns=["hour_minute_second", "person_ids", "agenders"])
    # df.hour_minute_second = hours
    # df.person_ids = ids
    # df.agenders = agenders

    # df.head()

    # if not(os.path.isdir(outputFolder)):
    #     os.mkdir(outputFolder)

    # df.to_csv(outputFolder + splitted_folder[-2] + ".csv")
    # beep = lambda x: os.system("echo -n '\a';sleep 0.4;" * x)
    # beep(10)

def compare_centroids(detections, agender, coords, img):
    # taken centroids
    taken = []
    info = []
    
    # iterate over agender centroids
    for detected in agender:
        if detected["region"]["x"] != 0 and detected["region"]["y"] != 0:
            face_centroid = (int(detected["region"]["x"]+(detected["region"]["w"]/2)),
                            int(detected["region"]["y"]+(detected["region"]["h"]/2)))

            print("face centroid: {} - box: {}".format(face_centroid, coords))

            # check if centroid is inside zone
            x, y = face_centroid 
            if not (x > coords["x1"] and x < coords["x2"] and y > coords["y1"] and y < coords["y2"]):
                print("face centroid outside boundaries...")
                continue

            min_distance = 10000
            min_centroid = None
            for indx, detection in enumerate(detections.xyxy):
                detection_centroid = (int((detection[0]+detection[2])/2), 
                                      int((detection[1]+detection[3])/2))
                
                dx, dy = detection_centroid

                # get distance between points
                distance = math.dist([x, y], [dx, dy])

                # update distance
                if (distance < min_distance) and (not indx in taken):
                    min_distance = distance
                    min_centroid = detection_centroid

                    taken.append(indx)
                    info.append({"id": detections.tracker_id[indx] ,"gender": detected["dominant_gender"], "age": detected["age"]})

            color_list = list(np.random.random(size=3) * 256)
            
            # draw centroids
            img = cv2.line(img, face_centroid, min_centroid, (255, 255, 255), 2)
            img = cv2.circle(img, face_centroid, 5, color_list, -1)
            img = cv2.circle(img, min_centroid, 5, color_list, -1)


            centroids_center = (int((x+dx)/2-20), int((y+dy)/2))
            img = cv2.putText(img, "{} - {} yo".format(detected["dominant_gender"], detected["age"]), centroids_center, cv2.FONT_HERSHEY_SIMPLEX , 0.7, color_list, 2, cv2.LINE_AA)

    return img, info



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "People counting and age+gender detection")
    parser.add_argument("-i",
                        "--imagesFolder",
                        help="Path to the folder where the input images files are stored",
                        type=str)
    parser.add_argument("-s",
                        "--store",
                        help="Store name to get restrictions",
                        type=str)
    parser.add_argument("-o",
                        "--outputFolder",
                        help="Path to the folder where to save the results",
                        type=str)
                        

    args = parser.parse_args()

    main(args.imagesFolder, args.store, args.outputFolder)
