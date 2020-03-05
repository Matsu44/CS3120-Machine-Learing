#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:08:06 2020

@author: fengjiang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'labelvalue']
# load dataset
pima = pd.read_csv("pima-indians-diabetes-database.csv", header=None, names=col_names)
print(pima.head())

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima['pregnant']
X2 = pima[['pregnant','insulin']] # Features
X3 = pima[0:3]
X4 =pima.loc[0:3,['pregnant','insulin']]  # 0:3 -> 4 rows
X5 = pima.iloc[0:3,0:1]  # 0:3 -> 3 rows, one column

y = pima.labelvalue # Target variable