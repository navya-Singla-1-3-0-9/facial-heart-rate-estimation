# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils
from matplotlib import pyplot as plt
import os
import random
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#@title
import tensorflow as tf
# Load the video file
import cv2

import os
import numpy as np
dirs = os.listdir('drive/My Drive')

def get_label(sub):
  gt = os.path.join(f'drive/My Drive/subject{sub}/ground_truth.txt')
  gtfilename = gt
  gtdata = np.loadtxt(gtfilename)
  gtTrace = gtdata[0,:].T
  gtTime = gtdata[2,:].T
  gtHR = gtdata[1,:].T
  return gtHR

def get_data(sub, off):
    images = []
    labels = []
  
    cap = cv2.VideoCapture(f"drive/My Drive/subject{sub}/vid.avi")
    # Loop through the video frames
    
    X_video = np.zeros((100, 64, 64, 3), dtype=np.uint8)
    y_video = np.zeros((100, 1))
    labels = get_label(sub)
    for i in range(off*100):
        ret, frame = cap.read()
    for i in range(100):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (64,64))
            X_video[i, :, :, :] = frame
            y_video[i, 0] = labels[off*100+i]
        else:
          break
    return X_video, y_video
import pandas as pd
def DenseToSparse(labels):
    labels = pd.Series(labels).apply(lambda c:(int)(c))
    labels = labels.astype(np.ubyte)
    sparse_labels = np.zeros((labels.shape[0], 200))
    sparse_labels[labels.index,labels] = 1
    return sparse_labels

images = []
labels = []
i=0
for sub in range(50):
    print("counter: ", sub)
    if(dirs.count(f'subject{sub}')==0):
      print(sub)
      continue
    for off in range(12):  
      img, label = get_data(sub, off)
      images.append(img)
      labels.append(label)
#sparse_labels = DenseToSparse(labels)

np.save('data_hr.npy', images)
np.save('labels_hr.npy', labels)

images = np.load('data_hr.npy')
labels = np.load('labels_hr.npy')

images = np.array(images)
labels = np.array(labels)

import tensorflow as tf

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', 
                                 input_shape=(100, 64, 64, 3)))
model.add(tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)))
model.add(tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[5]))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], X_test.shape[5]))

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=0, shuffle=True)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.20) # , random_state=40, shuffle=True)

#@title
save_path = '/content/model2.h5'
try:
    history = model.fit(x_train, y_train, epochs=300, validation_data=(x_valid, y_valid), batch_size = 32)
except KeyboardInterrupt:
    model.save(save_path)
    print('Output saved to: "{}./*"'.format(save_path))

model.evaluate(x_test, y_test, batch_size=1)

save_path = '/content/model2.h5'
try:
    history = model.fit(x_train, y_train, epochs=200, validation_data=(x_valid, y_valid), batch_size = 32)
except KeyboardInterrupt:
    model.save(save_path)
    print('Output saved to: "{}./*"'.format(save_path))

model.save(save_path)



import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()