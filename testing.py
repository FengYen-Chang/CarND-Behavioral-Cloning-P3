import csv
import cv2
import numpy as np
import sklearn
import random

def readSamples(csv_path) :
    labels = []
#     center_images = []
#     left_images = []
#     right_images = []
#     samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader :
            labels.append(line)
#             labels.append(line[3:])
#             center_images.append('../' + line[0][32:])
#             left_images.append('../' + line[1][32:])
#             right_images.append('../' + line[2][32:])
#             labels = line[3:]
#             center_images = line[0]
#             left_images = line[1]
#             right_images = line[2]
            
#             samples.append([center_images, left_images, right_images, labels])
            
    return labels#, center_images, left_images, right_images
#     return samples
if "__main__" :
    print (readSamples("./data/driving_log.csv")[0])