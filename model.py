from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda
from keras.applications.vgg16 import VGG16
from keras.layers import Cropping2D

from data_utils import readSamples, readSamples_ori, generator
from sklearn.model_selection import train_test_split

vgg_feature = VGG16(weights='imagenet', include_top=False)
for layers in vgg_feature.layers :
    layers.trainable = False
    
vgg_feature.summary()

# samples = readSamples("../track_1/driving_log.csv")
samples = readSamples_ori("./data/driving_log.csv")
samples = samples[1:]
print (samples[0])
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x : (x - 128.0) / 128.0, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(3,160,320)))
model.add(vgg_feature)
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
# model.add(Conv2D(128, (3, 3), input_shape=(5, 10, 256), padding='same'))
# model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

model.compile(loss = 'mse', optimizer='adam')
# model.fit_generator(train_generator, samples_per_epoch=len(train_samples) // 32, validation_data=validation_generator, validation_steps=len(validation_samples) // 32, nb_epoch=20)

import matplotlib.pyplot as plt
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples) // 32, validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples) // 32, 
    nb_epoch=20, verbose=1)

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
# model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2)

model.save('model_vgg_ori_20.h5')
exit()