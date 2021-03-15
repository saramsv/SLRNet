#! /usr/bin python
#run: python color_imgs_by_dir.py dir_name 
import numpy as np
import cv2
import os
import glob
import sys
import random



dir_name = sys.argv[1]

random.seed(0)
num_classes = 7
#colors = [(197, 215, 20), (132, 248, 207), (155, 244, 183), (111, 71, 144), (71, 48, 128), (75, 158, 50), (37, 169, 241)]
colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(num_classes)]
print(colors)
#colors = [(0, 0, 128), (0, 128, 0), (0, 128, 128), (128, 0, 0), (128, 0, 128), (128, 128, 0)]
#colors[0] = (0,0,0)
target_dir_name = dir_name.strip("/") + "_colored"
if not os.path.isdir(target_dir_name):
    os.mkdir(target_dir_name)
    print("mkdir " + target_dir_name)
else:
    print(target_dir_name + " exist")

for path in glob.glob(dir_name + "/"+ "*png"):
    img = cv2.imread(path) # this is the annotation
    for i in range(len(colors)):
        img[:,:,0][np.where(img[:,:,0] == i)] = colors[i][0]
        img[:,:,1][np.where(img[:,:,1] == i)] = colors[i][1]
        img[:,:,2][np.where(img[:,:,2] == i)] = colors[i][2]
    img_name = path.split('/')[-1]
    cv2.imwrite(target_dir_name + "/" + img_name, img)


