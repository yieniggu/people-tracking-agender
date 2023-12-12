import cv2
from deepface import DeepFace

img = cv2.imread("./data/pending/2023-10-20/16:50:07.jpg")

results = DeepFace.analyze(img, actions=("gender", "age", "race"), enforce_detection=False)

print(results)