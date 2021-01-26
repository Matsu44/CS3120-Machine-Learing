# -*- coding: utf-8 -*-
"""
Created on Thu May 13 09:38:21 2020

@author: simon
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
import os
import cv2
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

data_path = "../Data/CNN/animals/"

folder_list = os.listdir(data_path)
data = []
label = []
for folderName in folder_list:
    imagePath = data_path + folderName
    
    image_list = os.listdir(imagePath)
    for image_name in image_list:
        imageFullPath = data_path + folderName + "/" + image_name
       
        image = cv2.imread(imageFullPath)
        image = cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
        data.append(image)
        label.append(folderName)
        
        
    print ('subfolder done')
    
X = np.array(data)
y_prev = np.array(label)



le = preprocessing.LabelEncoder()
y = le.fit_transform(y_prev)

(trainX, testX, trainY, testY) = train_test_split(X,y,
                                 test_size=0.3,random_state=42)

trainX = trainX.reshape(trainX.shape[0], 32,32,3).astype('float')/255.0
testX = testX.reshape(testX.shape[0], 32,32,3).astype('float')/255.0


trainY = keras.utils.to_categorical(trainY, num_classes=3)
testY = keras.utils.to_categorical(testY, num_classes=3)

input_shape = (32,32,3)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),padding="same",
                 activation='relu',
                 input_shape=input_shape))
   
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5,5), padding="same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(3, activation='softmax'))

print("[INFO] training network...")
sgd = SGD(lr= 0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=1, batch_size=128)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),target_names = le.classes_))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 1), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 1), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 1), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 1), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()