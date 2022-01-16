import keras

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation, Conv2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, adadelta, adagrad, adam, adamax, rmsprop, nadam
from PIL import Image

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import time

from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

img_size = 100

categories = ['あ', 'い', 'う', 'え', 'お']

dense_size = len(categories)
print(dense_size)
X = []
Y = []

allfiles = []

for index, cat in enumerate(categories):
    print(index, cat)
    files = glob.glob('  ')
    print(files)
    for f in files:
        img = img_to_array(load_img(f , color_mode = 'rgb', target_size = (img_size, img_size)))

        X.append(img)
        Y.append(index)

X = np.asarray(X)
Y = np.asarray(Y)
X = X.astype('float32') / 255.0
Y = np_utils.to_categorical(Y, dense_size)

X_train. X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
print(Y_test)

model = Sequential()

model.add(Conv2D(100, (3, 3), padding = 'same', input_shape = X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(dense_size))
model.add(Activation('softmax'))

model.summary()

epochs = 100
batch_size = 100
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_test, Y_test))

plt.subplot(2, 1, 1)
plt.plot(range(epochs), history.history['acc'], label = 'acc')
plt.plot(range(epochs), history.history['val_acc'], label = 'val_acc')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))

plt.sunplot(2, 1, 2)
plt.plot(range(epochs), history.history['loss'], label = 'loss')
plt.plot(range(epochs), history.history['val_loss'], label = 'val_loss')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.show()

predoct_classes = model.predict_classes(X_test)
prob = model.predict_proba(X_test)

model.save('signLanguage.hdf5')

predict_classes = model.predict_classes(X_test, batch_size = 5)
true_classes = np.argmax(Y_test, 1)

print(confusion_matrix(true_classes, predict_classes))

print(model.evaluate(X_test, Y_test))