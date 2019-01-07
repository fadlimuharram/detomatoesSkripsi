#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:47:31 2018

@author: fadlimuharram
"""

from time import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import ZeroPadding2D
from keras.layers import Dropout
from keras.callbacks import TensorBoard

klasifikasi = Sequential()


klasifikasi.add(ZeroPadding2D((1,1),input_shape=(64, 64, 3)))
klasifikasi.add(Convolution2D(64, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(64, 3, 3, activation='relu'))
klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(128, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(128, 3, 3, activation='relu'))
klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(256, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(256, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(256, 3, 3, activation='relu'))
klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
klasifikasi.add(ZeroPadding2D((1,1)))
klasifikasi.add(Convolution2D(512, 3, 3, activation='relu'))
klasifikasi.add(MaxPooling2D((2,2), strides=(2,2)))

klasifikasi.add(Flatten())
klasifikasi.add(Dense(4096, activation='relu'))
klasifikasi.add(Dropout(0.5))
klasifikasi.add(Dense(4096, activation='relu'))
klasifikasi.add(Dropout(0.5))
klasifikasi.add(Dense(10, activation='softmax'))

# kompile cnn
klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
print("Compiling Initiated")

# part 2 -fitting the cnn to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        '/floyd/input/my_cool_first_cnn_aja_deh_dataset/color/training_set',
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '/floyd/input/my_cool_first_cnn_aja_deh_dataset/color/testing_set',
        target_size=(64, 64),
        batch_size=8,
        class_mode='categorical')

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

datanya = klasifikasi.fit_generator(
        train_set,
        steps_per_epoch=11307,
        epochs=10,
        validation_data=test_set,
        validation_steps=6854,
        callbacks=[tensorboard])

print("Compiling Completed")


# membuat prediksi
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('sp.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = klasifikasi.predict(test_image)
train_set.class_indices

print(datanya.history.keys())
print(datanya.history['acc'])
print(datanya.history['val_acc'])


import matplotlib.pyplot as plt

plt.plot(datanya.history['acc'])
plt.plot(datanya.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# history untuk loss
plt.plot(datanya.history['loss'])
plt.plot(datanya.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# simpan ke json
from keras.models import model_from_json

# serialize model ke JSON
model_json = klasifikasi.to_json()
with open("hasil/tomatVgg_classifier.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
klasifikasi.save_weights("hasil/tomatVgg_classifier.h5")
print("Saved model to disk")

# evaluate loaded model on test data
json_file = open('hasil/tomatVgg_classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_model_json)
# load weights into new model
loaded_classifier.load_weights("hasil/tomatVgg_classifier.h5")
print("Loaded model from disk")

