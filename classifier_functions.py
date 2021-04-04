import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import statistics
from sklearn.metrics import classification_report, confusion_matrix

from ast import literal_eval

import time
import datetime



import json

from random import sample
from random import seed

import tensorflow as tf

from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers
import os
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
#import tensorflow_addons as tfa

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers as reg

from keras.callbacks import ModelCheckpoint

from sklearn.utils import resample


metrics = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]
def build_model(optimizer = 'adam', loss='binary_crossentropy'):
    classifier = Sequential()
    classifier.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return classifier
    
def add_cnl(classifier, neurons, strd = (3, 3), p_size = (2,2), n = 0, drop = 0,pooling="max"):
    
    if n == 0:
        classifier.add(Conv2D(neurons[0], strd, input_shape = (150, 150, 1), activation = 'relu'))
        if pooling == max:
            classifier.add(MaxPooling2D(pool_size = p_size))
        else:
            classifier.add(AveragePooling2D(pool_size=p_size))
                           
        for i in range(1, len(neurons)):
            classifier.add(Conv2D(neurons[i], strd, activation = 'relu', padding='same'))
            if pooling == max:
                classifier.add(MaxPooling2D(pool_size = p_size))
            else:
                classifier.add(AveragePooling2D(pool_size=p_size))
            
    else:
        classifier.add(Conv2D(neurons, strd, input_shape = (150, 150, 1), activation = 'relu'))
        for i in range(1, n):
            classifier.add(Conv2D(neurons, strd, activation = 'relu'))
            if pooling == max:
                classifier.add(MaxPooling2D(pool_size = p_size))
            else:
                classifier.add(AveragePooling2D(pool_size=p_size))

    classifier.add(Flatten())
        
    if drop != 0:
        classifier.add(Dropout(drop))

def add_dense(classifier, neurons, n = 0, end = True, reg = None):
    if n == 0:
        for neu in neurons:
            classifier.add(Dense(neu,kernel_regularizer = reg, activation = 'relu'))

    else:
        for i in range(n):
            classifier.add(Dense(neurons,kernel_regularizer = reg ,activation = 'relu'))
    
    if end:
        classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Oversampling


def oversampling(traindf):
    covid_df_upsampled = resample(traindf.loc[traindf["class"]=="1"],
                          replace=True, # sample with replacement
                          n_samples=len(traindf.loc[traindf["class"]=="0"]), # match number in majority class
                          random_state=23) # reproducible results
    upsampled= pd.concat([covid_df_upsampled, traindf.loc[traindf["class"]=="0"]])
    return upsampled