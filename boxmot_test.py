import cv2
import argparse
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from stores_config import stores
import os
from boxmot import DeepOCSORT
import boto3
import pandas as pd
import json
import math

s3 = boto3.client("s3")
rekognition = boto3.client('rekognition')

def main(folder, store, outputFolder):
    unique_ids = []
    agenders = []
    faces_ids = []

    detection_results = []

    splitted_folder = folder.split("/")

    tracker = DeepOCSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
        device='cuda:0',
        fp16=False,
    )
    height = 480
    width = 720
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    store_set = stores[store]
    delim_area = store_set["coords"]

    images = sorted(os.listdir(args.imagesFolder))

    for image in images:
        img_ids = []

        img = cv2.imread(folder + image, cv2.IMREAD_COLOR)

        splitted_image = image.split(".")
        format_date = splitted_folder[-2] + " " + splitted_image[0].replace("-", ":")
        date = pd.to_datetime(format_date, format="%Y-%m-%d %H:%M:%S")


        img_rotated = cv2.resize(img, (height, width))

        if store_set["rotate"] is not None:
            img_rotated = cv2.rotate(img_rotated, store_set["rotate"])

        if store_set["crop"] is not None:
            img_rotated = img_rotated[store_set["crop"]["height"][0]:store_set["crop"]["height"][1] ,
                                      store_set["crop"]["width"][0]:store_set["crop"]["width"][1]]

        img_rotated_without_drawings = img_rotated.copy()
        # detect + track
        result = model.predict(img_rotated, conf=0.2, classes=0)[0]

        # take metadata from results object
        boxes = result.boxes  # Boxes object for bbox outputs
        xyxy = boxes.xyxy
        conf = boxes.conf
        xywh = boxes.xywh  # box with xywh format

        xyxy_detached = xyxy.detach().cpu().numpy()
        conf_detached = conf.detach().cpu().numpy()

        # print("xyxy detached: ", xyxy_detached)
        # print("conf detached: ", conf_detached)

        # draw delimited area
        cv2.rectangle(img_rotated, (delim_area["x1"], delim_area["y1"]),
                      (delim_area["x2"], delim_area["y2"]), (0, 255, 0), 2)

        results_for_tracker = []
        for coords, conf_score in zip(xyxy_detached, conf_detached):
            # print("coords: ", coords)
            # print("conf_score: ", conf_score)
            x1, y1, x2, y2 = coords
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # check if centroid of detection is inside the defined area
            x_centroid, y_centroid = (int((x1+x2)/2), int((y1+y2)/2))

            valid_x = x_centroid >= delim_area["x1"] and x_centroid <= delim_area["x2"]
            valid_y = y_centroid >= delim_area["y1"] and y_centroid <= delim_area["y2"]

            # print("bb centroid: ", (x_centroid, y_centroid))
            # print("valid_x: ", valid_x)
            # print("valid_y: ", valid_y)

            if not valid_x or not valid_y:
                print("bbox: ", coords, " outside restricted area")
                continue

            results_for_tracker.append([x1, y1, x2, y2, conf_score, 0])

            # Set color values for red, blue, and green
            # red_color = (0, 0, 255)  # (B, G, R)

            # cv2.circle(img_rotated, (x_centroid, y_centroid), 5, red_color, -1)
            # cv2.rectangle(img_rotated, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), red_color, 2)

            # text_color = (0, 0, 0)  # Black color for text
            # cv2.putText(img_rotated, "PERSON", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        if len(results_for_tracker) > 0:
            np_results_for_tracker = np.array(results_for_tracker)
        else:
            np_results_for_tracker = np.empty((0, 6))

        # print("np_results_for_tracker: ", np_results_for_tracker)
        tracks = tracker.update(np_results_for_tracker, img_rotated) # --> (x, y, x, y, id, conf, cls, ind)

        # print("tracks: ", tracks)

        if len(tracks) != 0:
            print("unique_ids: ", unique_ids)
            print("agenders: ", agenders)
            print("faces_ids: ", faces_ids)

            xyxys = tracks[:, 0:4].astype('int') # float64 to int
            ids = tracks[:, 4].astype('int') # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int') # float64 to int
            inds = tracks[:, 7].astype('int') # float64 to int

            # print bboxes with their associated id, cls and conf
            if tracks.shape[0] != 0:
                for xyxy, id, conf in zip(xyxys, ids, confs):
                    img_rotated = cv2.rectangle(
                        img_rotated,
                        (xyxy[0], xyxy[1]),
                        (xyxy[2], xyxy[3]),
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        img_rotated,
                        f'id: {id}',
                        (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

                    img_ids.append(id)

                # if at least one element of image ids is not in unique ids
                # get faces of image
                if any(x not in faces_ids for x in img_ids):
                    print("new img_id without face| i{} - f{}".format(img_ids, faces_ids))
                    bucket_path = "{}/{}/{}".format(store, splitted_folder[-2], image)
                    # print("bucket_path: ", bucket_path)

                    image_bytes = cv2.imencode('.jpg', img_rotated_without_drawings)[1].tobytes()
                    # s3.put_object(Bucket="subway-persons", Key=bucket_path, Body=image_bytes)

                    # call facial detect to get age+gender on aws rekognition
                    face_detections = detect_faces(bucket_path, image_bytes)

                    # associate detected faces to the id's
                    tracker_data = {"detections": xyxys, "ids": ids}
                    img_rotated, unique_ids, agenders, faces_ids = compare_centroids(tracker_data, face_detections["FaceDetails"],
                                                                            unique_ids, agenders, delim_area,
                                                                            faces_ids, img_rotated)


                    # draw face bounding box
                    #img_rotated = draw_face_bounding_box(face_detections, img_rotated)

        detection_results.append([date, img_ids])


        # show image with bboxes, ids, classes and confidences
        # cv2.imshow('frame', img_rotated)

        cv2.imwrite("data/results/temp/merced/{}".format(image), img_rotated)

        # # break on pressing q
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break


    # print("total of: ", len(unique_ids)," unique persons detected in delimited area")
    # # print("unique_ids: ", unique_ids)

    # # build results df
    # ids_df = pd.DataFrame(detection_results, columns=["hour_minute_second", "persons"])
    # agenders_df = pd.DataFrame(agenders, columns=["person_id", "gender", "age_range"])

    # if not(os.path.isdir(outputFolder)):
    #     os.mkdir(outputFolder)

    # ids_df.to_csv(outputFolder + splitted_folder[-2] + "_ids.csv", index=False)
    # agenders_df.to_csv(outputFolder + splitted_folder[-2] + "_agenders.csv", index=False)

    # with pd.ExcelWriter(outputFolder + splitted_folder[-2] + ".xlsx") as writer:
    #     ids_df.to_excel(writer, sheet_name="ids", index=False)
    #     agenders_df.to_excel(writer, sheet_name="age_gender", index=False)


    # os.system("beep -f 2000 -l 1500")
    # cv2.destroyAllWindows()

def draw_face_bounding_box(detections, img):
    height, width, _ = img.shape

    for detection in detections["FaceDetails"]:
        bounding_box = detection["BoundingBox"]

        left = width*bounding_box["Left"]
        top = height*bounding_box["Top"]
        box_width = width*bounding_box["Width"]
        box_height = height*bounding_box["Height"]

        x1 = int(left)
        y1 = int(top)
        x2 = int(left+box_width)
        y2 = int(top+box_height)

        img = cv2.rectangle(img, (x1, y1),
                      (x2, y2), (255, 0, 0), 2)

    return img

def detect_faces(photo, img):
    response = rekognition.detect_faces(Image={'Bytes': img},
                                   Attributes=['ALL'])

    # print('Detected faces for ' + photo)
    # for faceDetail in response['FaceDetails']:
    #     print('The detected face is between ' + str(faceDetail['AgeRange']['Low'])
    #           + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')

    #     print('Here are the other attributes:')
    #     print(json.dumps(faceDetail, indent=4, sort_keys=True))

    #     # Access predictions for individual face details and print them
    #     print("Gender: " + str(faceDetail['Gender']))
    #     print("Smile: " + str(faceDetail['Smile']))
    #     print("Eyeglasses: " + str(faceDetail['Eyeglasses']))
    #     print("Face Occluded: " + str(faceDetail['FaceOccluded']))
    #     print("Emotions: " + str(faceDetail['Emotions'][0]))

    return response

def compare_centroids(tracker_data, face_details, unique_ids, agenders, delim_area, faces_ids, img):
    height, width, _ = img.shape

    faces_matched = {}
    pending_matches = list(range(len(tracker_data["detections"])))

    print("pending_matches: ", pending_matches)
    # iterate over agender centroids
    for face_index, face in enumerate(face_details):
        bounding_box = face["BoundingBox"]

        left = width*bounding_box["Left"]
        top = height*bounding_box["Top"]
        box_width = width*bounding_box["Width"]
        box_height = height*bounding_box["Height"]

        # get the centroid of the face bounding box
        face_centroid = (int(left+(box_width/2)),
                            int(top+(box_height/2)))

        # print("face centroid: {}".format(face_centroid))

        # check if centroid is inside zone
        x, y = face_centroid
        valid_x = x >= delim_area["x1"] and x <= delim_area["x2"]
        valid_y = y >= delim_area["y1"] and y <= delim_area["y2"]

        if(valid_x and valid_y):
            faces_matched[face_index] = {"face_centroid": face_centroid,
                                         "gender": face["Gender"]["Value"],
                                         "age_range": "[{}-{}]".format(face["AgeRange"]["Low"], face["AgeRange"]["High"]),
                                         "emotion": face["Emotions"][0]["Type"]}


    print("========= INITIAL FACES MATCHED ==========")
    print("initial faces_matched: ", faces_matched)
    print("========= INITIAL FACES MATCHED ==========")


    # loop until are faces are matched
    detection_retrys = {}
    while len(pending_matches) != 0:
        for detection_index in pending_matches:
            detection_matched = False
            print("========= PROCESSING NEW DETECTION ==========")
            print("id: {} - detection_index: {}".format(tracker_data["ids"][detection_index], detection_index))
            print("========= PROCESSING NEW DETECTION ==========")

            # check for retrys flag
            retrys = detection_retrys[detection_index] if detection_index in detection_retrys else 1

            print("retrys for {}: {}".format(detection_index, retrys))
            if retrys > 1:
                index_of = pending_matches.index(detection_index)
                pending_matches.pop(index_of)

                continue

            detection = tracker_data["detections"][detection_index]
            # detection_centroid = (int((detection[0]+detection[2])/2),
            #                         int((detection[1]+detection[3])/2))

            detection_upper_centroid = (int((detection[0]+detection[2])/2),
                                    int((detection[1])))

            dx, dy = detection_upper_centroid

            if tracker_data["ids"][detection_index] not in unique_ids:
                unique_ids.append(tracker_data["ids"][detection_index])


            # iterate over faces
            for face_key in faces_matched:
                face_x, face_y = faces_matched[face_key]["face_centroid"]
                distance = math.dist([face_x, face_y], [dx, dy])


                # print("current_face: ", faces_matched["face_key"])
                # # check if detection has a match
                existing_match = faces_matched[face_key].get("detection_index")
                min_distance = faces_matched[face_key]["min_distance"] if "min_distance" in faces_matched[face_key].keys() else 10000

                print("========== FACE MATCH ATTEMPT ============")
                print("current_face: ", faces_matched[face_key])
                print("distance: ", distance)
                print("existing_match: ", existing_match)
                print("current min distance: ", min_distance)
                print("========== FACE MATCH ATTEMPT ============")

                # update distance
                # check also if face centroid is inside detection boundaries
                print("face_x: {} - x1: {} - x2: {}".format(face_x, detection[0], detection[2]))
                print("face_y: {} - y1: {} - y2: {}".format(face_y, detection[1], detection[3]))
                valid_face_x = face_x >= detection[0] and face_x <= detection[2]
                valid_face_y = face_y >= detection[1] and face_y <= detection[3]

                # print("valid_X: {} - valid_y: {}".format(valid_x, valid_y))

                if (distance < min_distance and valid_face_x and valid_face_y and not detection_matched):
                    print("========== FACE MATCH SUCCESS ============")
                    print("detection_index: ", detection_index)
                    print("id: ", tracker_data["ids"][detection_index])
                    print("face: ", faces_matched[face_key])
                    print("========== FACE MATCH SUCCESS ============")

                    # print("distance of association: ", distance)
                    # print("x_distance: ", x_distance)
                    # print("y_distance: ", y_distance)
                    # print("current detection index: ", detection_index)

                    # add detection to matches
                    faces_matched[face_key]["detection_index"] = detection_index
                    faces_matched[face_key]["min_distance"] = distance
                    faces_matched[face_key]["min_centroid"] = detection_upper_centroid

                    # if match already re add detection to unmatched
                    if existing_match is not None:
                        del detection_retrys[existing_match]
                        pending_matches.append(existing_match)
                        print("========== EXISTING MATCH ============")
                        print("existing_match: ", existing_match)
                        print("pending_matches: ", pending_matches)
                        print("========== FACE MATCH SUCCESS ============")

                    detection_matched = True

            # remove matched detection
            if detection_matched:
                print("pending matches before index_of: ", pending_matches)
                index_of = pending_matches.index(detection_index)
                pending_matches.pop(index_of)

            # check for retrys flag
            detection_retrys[detection_index] = retrys + 1

    # draw faces matched
    for face_key in faces_matched:
        face_matched = faces_matched[face_key]

        detection_index = face_matched.get("detection_index")
        id_of_detection = tracker_data["ids"][detection_index]
        # check if key is empty
        if detection_index is None or id_of_detection in faces_ids:
            print("========== DETECTION INDEX NOT FOUND ============")
            print("detection_index: ", detection_index)
            print("id of detection: ", id_of_detection)
            print("face: ", face_matched)
            print("========== DETECTION INDEX NOT FOUND ============")
            continue

        # add the detection to matched faces
        if (tracker_data["ids"][detection_index] not in faces_ids):
            print("========== ADDING NEW ID TO FACES IDS ============")
            print("adding new match face with id: ", id_of_detection)
            print("faces_ids: ", faces_ids)
            print("========== ADDING NEW ID TO FACES IDS ============")

            agenders.append([tracker_data["ids"][face_matched["detection_index"]],
                            face_matched["gender"],
                            face_matched["age_range"],
                            face_matched["emotion"]])

            faces_ids.append(tracker_data["ids"][face_matched["detection_index"]])

        color_list = list(np.random.random(size=3) * 256)

        # draw centroids
        img = cv2.line(img, face_matched["face_centroid"], face_matched["min_centroid"], (255, 255, 255), 2)
        img = cv2.circle(img, face_matched["face_centroid"], 5, color_list, -1)
        img = cv2.circle(img, face_matched["min_centroid"], 5, color_list, -1)

        id = tracker_data["ids"][detection_index]
        img = cv2.putText(img, "id:{} | {} | {} | age: {}".format(id, face_matched["gender"], face_matched["emotion"], face_matched["age_range"]),
                                                (5, 17+(face_key*17)), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    return img, unique_ids, agenders, faces_ids

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