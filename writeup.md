# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figure_1.png "loss of final architecture"
[image2]: ./figure_2.png "loss of second last architecture"

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