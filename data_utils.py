import csv
import cv2
import numpy as np
import sklearn
import random


def readSamples(csv_path) :
#     labels = []
#     center_images = []
#     left_images = []
#     right_images = []
    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader :
#             labels.append(line)
#             labels.append(line[3:])
#             center_images.append('../' + line[0][32:])
#             left_images.append('../' + line[1][32:])
#             right_images.append('../' + line[2][32:])
            labels = line[3:]
            center_images = '../' + line[0][32:]
            left_images = '../' + line[1][32:]
            right_images = '../' + line[2][32:]
            
            samples.append([center_images, left_images, right_images, labels])
            
#     return labels, center_images, left_images, right_images
    return samples

def readSamples_ori(csv_path) :
#     labels = []
#     center_images = []
#     left_images = []
#     right_images = []
    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader :
#             labels.append(line)
#             labels.append(line[3:])
#             center_images.append('../' + line[0][32:])
#             left_images.append('../' + line[1][32:])
#             right_images.append('../' + line[2][32:])
            labels = line[3:]
            center_images = './data/' + line[0]
            left_images = './data/' + line[1]
            right_images = './data/' + line[2]
            
            samples.append([center_images, left_images, right_images, labels])
            
#     return labels, center_images, left_images, right_images
    return samples

def imageReader(imgPath) :
    return cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

def generator(samples, batch_size = 32) :
#     samples = readLabels(csv_path)
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
#             labels = []
            for batch_sample in batch_samples:
                img_path = batch_sample[0]
                center_image = imageReader(img_path)
                center_angle = batch_sample[3][0]
                label = batch_sample[3][:]
                images.append(center_image)
                angles.append(center_angle)
#                 labels.append(label)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles, dtype = np.float32)
#             y_train = np.array(labels, dtype = np.float32)
            yield sklearn.utils.shuffle(X_train, y_train)

# if "__main__" :
#     print (readLabels("../track_1/driving_log.csv")[0][0][32:])
#     labels, center_images, left_images, right_images = readLabels("../track_1/driving_log.csv")
#     samples = readLabels("../track_1/driving_log.csv")
#     print (samples[0])
#     print (len(samples))
#     print (right_images[0])


