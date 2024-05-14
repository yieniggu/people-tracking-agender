from ultralytics import YOLO
import cv2
import pandas as pd
import os
import argparse
from stores_config import stores
import sys
import math
import numpy as np

def main(folder, store):
    model = YOLO("yolov8n.pt")  # load

    height = 480
    width = 720

    store_set = stores[store]
    print("store: ", store_set)

    images = sorted(os.listdir(args.imagesFolder))
    # print("images: ", images)

    video = cv2.VideoWriter("test.avi", 0, 1, (width, height))

    for image in images:
        img = cv2.imread(folder + image, cv2.IMREAD_COLOR)

        img_rotated = cv2.resize(img, (height, width))

        if store_set["rotate"] is not None:
            img_rotated = cv2.rotate(img_rotated, store_set["rotate"])

        if store_set["crop"] is not None:
            img_rotated = img_rotated[store_set["crop"]["height"][0]:store_set["crop"]["height"][1] ,
                                      store_set["crop"]["width"][0]:store_set["crop"]["width"][1]]

        video.write(img_rotated)

    # if not(os.path.isdir(outputFolder)):
    #     os.mkdir(outputFolder)



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

    args = parser.parse_args()

    main(args.imagesFolder, args.store)
