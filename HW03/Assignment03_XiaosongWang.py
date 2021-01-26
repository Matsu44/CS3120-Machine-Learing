# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:49:49 2020

@author: simon
"""

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import operator

data_path = "../Data/KNN/animals/"

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
    
finalData = np.array(data)
finalLabel = np.array(label)

finalData = finalData.reshape(finalData.shape[0],3072)


le = preprocessing.LabelEncoder()
mylabels = le.fit_transform(finalLabel)

(trainAllX, testX, trainAllY, testY) = train_test_split(finalData,mylabels,
                                 test_size=0.2,random_state=42)
(trainX, validationX, trainY, validationY) = train_test_split(trainAllX,trainAllY,
                                 test_size=0.125,random_state=42)
print ('Train Test Validate Split Done')


f_score = []

for k in [3, 5, 7]:
    for MinkowskiMetric in [1, 2]:
        print('k:' + str(k) + ' Distance: L' + str(MinkowskiMetric))
        model = KNeighborsClassifier(n_neighbors=k, p = MinkowskiMetric)
        model.fit(trainX,trainY)
        predY = model.predict(validationX)
        print (classification_report(validationY,predY,target_names = le.classes_))
        f1_score = metrics.f1_score(validationY,predY,labels=None,pos_label=1,average='macro',sample_weight=None)
        
        # Save all f1_score into a list
        f_score.append(f1_score)
        
print (f_score)

# Find the maximum of f_score and get its index
index, f_score_max =  max(enumerate(f_score), key=operator.itemgetter(1))
print('The highest f score is ' + str(f_score_max))

# Select the best k and p
list_k_p = [[3,1],[3,2],[5,1],[5,2],[7,1],[7,2]]
k_p = list_k_p[index]
k = k_p[0]
MinkowskiMetric = k_p[1]
print('The best k is ' + str(k))
print('The best L is ' + str(MinkowskiMetric))

# Testing
print('Testing:')
model = KNeighborsClassifier(n_neighbors=k, p = MinkowskiMetric)
model.fit(trainX,trainY)
predY = model.predict(testX)
print (classification_report(testY,predY,target_names = le.classes_))