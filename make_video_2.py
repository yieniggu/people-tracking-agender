import cv2
import os

#TODO
image_folder = 'data/results/temp/merced/'
video_name = 'merced.avi'#save as .avi
#is changeable but maintain same h&w over all  frames

width=480
height=720
#this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video=cv2.VideoWriter(video_name,fourcc, 2.0, (width,height))

images = sorted(os.listdir(image_folder))

print("images: ", images)

for i in (images):
     print("processing image: ", i)
     x=cv2.imread(image_folder+i, cv2.IMREAD_COLOR)
     video.write(x)


cv2.destroyAllWindows()
video.release()

