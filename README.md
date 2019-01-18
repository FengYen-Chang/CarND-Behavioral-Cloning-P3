# Behavioral Cloning Project
The complete project of Self-Driving Car Engineer Nanodegree Program. I done the assignment at winter, 2018. I just upload this in to github.

# Model
You can download the variable data via [here](https://drive.google.com/file/d/1ovxcRl95gj5PhG-CwiFns2Pl_5Wmt8Sn/view?usp=sharing).

# The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figure_1.png "loss of final architecture"
[image2]: ./figure_2.png "loss of second last architecture"

# Usage
## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

# Dataset

Collect dataset by yourself or download the dataset from Udacity. In this project, I was used the dataset which provided by Udacity.

## Rubric Points

---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to let model how to drive the car in simulator as me. However, I can't achieve this point since my computer was too slow can't drive car will. To solve this problem, I was trained the network via training data which provide by udacity.

In this project, I was using VGG16 as feature extractor because I just want to get the feature map from training images and doesn't want the image resized by some factor before training process. Before feature extractor, I normailze the input data from [0 ~ 255] to [-1 ~ 1], then I crop the image to reduce the dimension of input.

In addition, I added one convnet after vgg feature extractor not only reduce the feature dimension, but also let this convnet learning more useful thing for drive the car on the simulator since the vgg feature extractor was trained by imagenet dataset, not for car driving. Then I flatten the feature map and connect the fc layer as output.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I added the another fc layer to make it work well. Additionnaly reduce the epoch on training process.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 27-39) consisted of a convolution neural network with the following layers and layer sizes after vgg feature extractor, the convnet have 3-by-3 kernel and 256 filters, and two fc-layers one is 128, another one is 1. The model architecture was listed on the below.

##### Model Architecture :

* image normalize
* image crop
* VGG16 feature extractor
* * Conv. Net 3x3 kernel, 64 filters
* * Conv. Net 3x3 kernel, 64 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 128 filters
* * Conv. Net 3x3 kernel, 128 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 256 filters
* * Conv. Net 3x3 kernel, 256 filters
* * Conv. Net 3x3 kernel, 256 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Max pooling 2D
* Conv. Net 3x3 kernel, 256 filters
* FC layers 128 dimension
* FC layers 1 dimension


#### 3. Creation of the Training Process

In this project, I use the data set which provide by udacity to train the network and randomly shuffled the data set then put 20% of the data into a validation set. For load data, I wrote the `data_utils.py` to parse the image directory and labels, meanwhile, construct the generator to provide data in training process.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary and training the model by 20 epoch.

#### 4. Another Model Architecture

Before I got the final model, I was try to train the many architecture, such as using two convnet after vgg fearure extractor but it doesn't work well. In here I want to show the second last model I trained, and the architecture of this model was listed on the below.
* image normalize
* image crop
* VGG16 feature extractor
* * Conv. Net 3x3 kernel, 64 filters
* * Conv. Net 3x3 kernel, 64 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 128 filters
* * Conv. Net 3x3 kernel, 128 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 256 filters
* * Conv. Net 3x3 kernel, 256 filters
* * Conv. Net 3x3 kernel, 256 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Max pooling 2D
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Conv. Net 3x3 kernel, 512 filters
* * Max pooling 2D
* Conv. Net 3x3 kernel, 256 filters
* FC layers 1 dimension

The second last model was trained by 30 epoch, this model was work great on first lape, but it have some problem on second lape. To solve it I try the last architecture and trained by 20 epoch because I think it could converge before 20 epoch, and the loss figures of the two model was shown on below.

* Loss changes of final architecture
![alt text][image1]

* Loss changes of second last architecture
![alt text][image2]

Additionally, I think the dimension of flatten feature is too large it may influence the result easily, so I add another one fc layer before output layer, and it make the result well.
