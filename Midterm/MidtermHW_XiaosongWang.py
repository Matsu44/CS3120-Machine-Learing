# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:28:35 2020

@author: simon
"""

import numpy as np
import sklearn.svm as svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Read data
data_path = "../Data/train.csv"
data = np.array(pd.read_csv(data_path))
print("[INFO] evaluating classifier...")

# I just use the small piece data for training and testing. I've tried to use all, but it last forever
X1 = data[:1000 , 1:]
y1 = data[:1000 , 0]
# Different size of data
X2 = data[:2000 , 1:]
y2 = data[:2000 , 0]
# The same size of data but different sample
X3 = data[-2000: , 1:]
y3 = data[-2000: , 0]

# Split training and testing dataset for Lofistic Regression and Decision Tree
trainX1, testX1, trainY1, testY1 = train_test_split(X1, y1, test_size = 0.3, random_state = 0)
trainX2, testX2, trainY2, testY2 = train_test_split(X2, y2, test_size = 0.3, random_state = 0)
trainX3, testX3, trainY3, testY3 = train_test_split(X3, y3, test_size = 0.3, random_state = 0)

# Logistic Regression Classifier
model1 = LogisticRegression()
model1.fit(trainX1, trainY1)
predY=model1.predict(testX1)
print("Logistic Regression Classification Report -1")
print(classification_report(testY1, predY)) 

model1 = LogisticRegression()
model1.fit(trainX2, trainY2)
predY=model1.predict(testX2)
print("Logistic Regression Classification Report -2")
print(classification_report(testY2, predY)) 

model1 = LogisticRegression()
model1.fit(trainX3, trainY3)
predY=model1.predict(testX3)
print("Logistic Regression Classification Report -3")
print(classification_report(testY3, predY)) 

# Decision Tree Classifier
model2 = DecisionTreeClassifier()
model2.fit(trainX1, trainY1)
predY=model2.predict(testX1)
print("Decision Tree Classification Report -1")
print(classification_report(testY1, predY)) 

model2 = DecisionTreeClassifier()
model2.fit(trainX2, trainY2)
predY=model2.predict(testX2)
print("Decision Tree Classification Report -2")
print(classification_report(testY2, predY)) 

model2 = DecisionTreeClassifier()
model2.fit(trainX3, trainY3)
predY=model2.predict(testX3)
print("Decision Tree Classification Report -3")
print(classification_report(testY3, predY))

# Split training, validation, and testing dataset for svm classifier
(trainAllX, testX, trainAllY, testY) = train_test_split(X1,y1, test_size=0.2,random_state=42)
(trainX, validationX, trainY, validationY) = train_test_split(trainAllX,trainAllY, test_size=0.125,random_state=42)

# Using validation dataset for choosing the best performance kernel. Other hyperparameter, such as C and gamma, are the same
C = 1
gamma = 0.001
model3 = svm.SVC(kernel = 'linear', C = C, gamma = gamma)
model3.fit(trainX, trainY)
predY=model3.predict(validationX)
print("Linear SVM Validation Report")
print(classification_report(validationY, predY)) 

C = 1
gamma = 0.001
model3 = svm.SVC(kernel = 'poly', C = C, gamma = gamma)
model3.fit(trainX, trainY)
predY=model3.predict(validationX)
print("Poly SVM Validation Report")
print(classification_report(validationY, predY))

C = 1
gamma = 0.001
model3 = svm.SVC(kernel = 'rbf', C = C, gamma = gamma)
model3.fit(trainX, trainY)
predY=model3.predict(validationX)
print("RBF SVM Validation Report")
print(classification_report(validationY, predY))

# The poly shows the best performance. Then do the testing
C = 1
gamma = 0.001
model3 = svm.SVC(kernel = 'poly', C = C, gamma = gamma)
model3.fit(trainX, trainY)
predY=model3.predict(testX)
print("SVM Classification Report")
print(classification_report(testY, predY))


