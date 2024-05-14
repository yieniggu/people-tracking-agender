from ultralytics import YOLO
import cv2
import supervision as sv
import pandas as pd
import os
import argparse
from stores_config import stores
import sys
import math
import numpy as np

def main(folder, store, outputFolder):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    byte_tracker = sv.ByteTrack()

    unique_ids = []
    info = []

    # init dataframe data
    hours = []
    ids = []
    agenders = []

    height = 480
    width = 720

    store_set = stores[store]
    print("store: ", store_set)

    zone = sv.PolygonZone(store_set["polygon"], frame_resolution_wh=(width, height), triggering_position=sv.Position.CENTER)

    images = sorted(os.listdir(args.imagesFolder))
    # print("images: ", images)

    for image in images:
        img = cv2.imread(folder + image, cv2.IMREAD_COLOR)

        img_rotated = cv2.resize(img, (height, width))

        if store_set["rotate"] is not None:
            img_rotated = cv2.rotate(img_rotated, store_set["rotate"])

        if store_set["crop"] is not None:
            img_rotated = img_rotated[store_set["crop"]["height"][0]:store_set["crop"]["height"][1] ,
                                      store_set["crop"]["width"][0]:store_set["crop"]["width"][1]]

        # detect + track
        results = model(img_rotated)[0]

        #print("yolo results: ", result)
        #print("#"*100)
        detections = sv.Detections.from_ultralytics(results)
        mask = zone.trigger(detections=detections)
        detections = detections[(detections.class_id == 0) & mask]

        detections = byte_tracker.update_with_detections(detections)
        print("detections: ", detections)

        # if(detections.__len__() > 0):
        #     agender = DeepFace.analyze(img_rotated, actions=("gender", "age"), enforce_detection=False)
        #     print("agender: ", agender)

        #     img_rotated, info = compare_centroids(detections, agender, store_set["coords"], img_rotated)

        print("#"*100)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
        label_annotator = sv.LabelAnnotator()

        frame = box_annotator.annotate(
            scene=img_rotated, detections=detections, labels=labels
        )



        color = sv.Color(150, 200, 80)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=color ,thickness=2)
        frame = zone_annotator.annotate(frame)

        # add to results
        splitted_image = image.split(".")
        splitted_folder = folder.split("/")
        print("splitted_folder: ", splitted_folder)
        print("tracker ids: ", ids)
        format_date = splitted_folder[-2] + " " + splitted_image[0].replace("-", ":")
        date = pd.to_datetime(format_date, format="%Y-%m-%d %H:%M:%S")
        hours.append(date)
        ids.append(detections.tracker_id)
        # agenders.append(info)

        # cv2.imwrite(outputFolder + "results/"+ image, frame)

        cv2.imshow("yolov8", frame)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            sys.exit()


    print("Generating dataframe...")
    df = pd.DataFrame(columns=["hour_minute_second", "person_ids", "agenders"])
    df.hour_minute_second = hours
    df.person_ids = ids
    # df.agenders = agenders

    df.head()

    if not(os.path.isdir(outputFolder)):
        os.mkdir(outputFolder)

    df.to_csv(outputFolder + splitted_folder[-2] + ".csv")

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

def upload_to_s3():

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
