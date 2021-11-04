# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:38:59 2020

@author: abhijith
""" 
# organize imports
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
# for machine learning we cannot have any data
# or text so we conver it to numbers hance used Label Encoder
from sklearn.preprocessing import MinMaxScaler
#gets feature within a range
import numpy as np
import joblib
#numerical
import mahotas
# Computer vision toolbox contains algorithm
#for feature extraction in general and its works faster
import cv2
#open CV library
import os
#Operating system
import h5py
#let you store huge amounts of numerical data,
#and easily manipulate that data from NumPy
from pathlib import Path
#handle filesystem paths
import json
#to save in json format
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
#from dbn.tensorflow import SupervisedDBNClassification
fixed_size=tuple((256,256))
file='F:\\Desktop\\SKIN DNN Final\\Database\\diseased\\ISIC_0000147.jpg'
bins=8
#%% FEATURE EXTRACTION
#Hu moments
# feature descriptor-1: Hu Moments 7 feature
def fd_hu_moments(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    feature=cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
#feature descriptor-1:
def fd_haralick(image): #glcm feature
#convert the image to grayscale
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# compute the haralick texture feature vector
    haralick=mahotas.features.haralick(gray).mean(axis=0)
#return the result
    return haralick

#feature-descriptor-3: Color Histogram

def fd_histogram(image,mask=None):
    #convert the image to HSV color-space
    image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #compute the color histogram
    hist=cv2.calcHist([image],[0,1,2],None,[bins,bins,bins],
                      [0,256,0,256,0,256])
    #normalize the histogram
    cv2.normalize(hist,hist)
    #return the histogram
    return hist.flatten()


image= cv2.imread(file) #read image
image=cv2.resize(image,fixed_size) #resize image
fv_hu_moments=fd_hu_moments(image)
fv_haralick=fd_haralick(image)
fv_histogram=fd_histogram(image)
global_feature=np.hstack([fv_histogram,fv_haralick,fv_hu_moments])
global_feature=np.reshape(global_feature,(1, -1))
x = pd.DataFrame(global_feature)
X=x.iloc[:,0:15]
model=joblib.load('dnn.pk1')
preds = model.predict_classes(np.reshape(X,(1, -1)), verbose=0)
print(preds) 
if preds[0]==0:
    print('Diseased')
else:
    print('Not Diseased')