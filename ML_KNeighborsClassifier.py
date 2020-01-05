#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:32:55 2020

@author: salama
"""

import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,classification_report,recall_score
from sklearn.preprocessing import LabelBinarizer,LabelEncoder,StandardScaler, 

#--------------------------------------------#
#THIS TO READ CSV FILE#
data = pd.read_csv('diabetes.csv')
#-------------------------------------------#
#INFORMATION ABOUT DATASET#
data.shape
data.isnull().sum()
data.describe()
data.info()
data['Diabetes'].value_counts()

#-------------------------------------------#
#TO  lABEL CODE COTEGRIES#
lab = LabelEncoder()
data.Diabetes  = lab.fit_transform(data.Diabetes)
data.head(2)

'''or conver number to text
data['Diabetes'] = data['Diabetes'].map({0:'no', 1:'yes'})
'''
#=================================================================#
''' HOW TO MADE FEATURE AND TARGET'''
X,Y = data.drop(['Diabetes'], axis=1).values, data['Diabetes'].values

'''TO TRAINNING DATA'''
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=0)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_prod = knn.predict(x_test)


def modelClassifierVisulation():
    model = []
    model.append(('THIS SCORE :{}'.format(knn.score(x_test,y_test))))
    model.append(('THIS ACCURACY:{}'.format(accuracy_score(y_test,y_prod))))
    model.append(('CONFUSION:{}'.format(confusion_matrix(y_test,y_prod))))
    model.append(('REPORT:{}'.format(classification_report(y_test,y_prod))))
    model.append(('REPORT:{}'.format(recall_score(y_test,y_prod))))
    for i in model:
        print(i)
modelClassifierVisulation()        
''' how to calculter matrie168+50+37+53 / 168+50 '''



'''------------------------------------------------------------------'''
from sklearn.model_selection import cross_val_score
#create new knn model#
cv_knn = KNeighborsClassifier(n_neighbors=3)
cv_score = cross_val_score(cv_knn, X,Y, cv=5)

print('CV_SCORES MEAN :()'.format(np.mean(cv_score)))
print(cv_score)















