from ultralytics import YOLO
import cv2
import os
import argparse
import sys
from stores_config import stores

def main(folder, store):
    images = sorted(os.listdir(args.imagesFolder))
    # print("images: ", images)

    height = 480
    width = 720

    store_set = stores[store]
    print("store: ", store_set)

    for image in images:
        img = cv2.imread(folder + image, cv2.IMREAD_COLOR)

        img_rotated = cv2.resize(img, (height, width))

        if store_set["rotate"] is not None:
            img_rotated = cv2.rotate(img_rotated, store_set["rotate"])
        

        cv2.imshow("yolov8", img_rotated)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            sys.exit()



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
