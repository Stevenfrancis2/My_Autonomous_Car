import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import tensorflow as tf

from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax

import matplotlib.image as mpimg
from imgaug import augmenters as iaa

import random


#### STEP 1 - INITIALIZE DATA
def getName(filePath):
    return filePath.split('\\')[-1]

def getSteeringInMatrix(steering):

    cmd=steering
    if(cmd == "b'fl'"):
        return[1,0,0]
    elif(cmd =="b'f'"):
        return [0,1,0]
    elif(cmd=="b'fr'"):
        return [0,0,1]
    else:
        return [1,1,1]
def getSteeringInAngle(steering):
    cmd = steering
    if (cmd == "b'fl'"):
        str=float(-0.39)
        return str
    elif (cmd == "b'f'"):
        str=float(0)
        return str
    elif (cmd == "b'fr'"):
        str=float(0.39)
        return str
    else:
        str=2
        return str



def importDataInfo(path):
    columns = ['Center','Steering']
    noOfFolders = len(os.listdir(path))//2
    data = pd.DataFrame()


    dataNew = pd.read_csv(os.path.join(path, 'log_0.csv'), names = columns)
    print(f'log_0.csv:{dataNew.shape[0]} ',end='')
    #### REMOVE FILE PATH AND GET ONLY FILE NAME
    #print(getName(data['Center'][0]))
    dataNew['Center']=dataNew['Center'].apply(getName)
    data =data.append(dataNew,True )

    print(' ')
    print('Total Images Imported',data.shape[0])
    return data

def removeNeutralMatrix(data):
    print('loaded before delete Images:', len(data))
    removeIndexlist=[]
    for i in range(len(data['Steering'])):
        if data['Steering'][i]==[1,1,1]:
            removeIndexlist.append(i)

    print('Removed Images:', len(removeIndexlist))
    data.drop(data.index[removeIndexlist], inplace=True)
    print('Remaining Images:', len(data))
    return data
def removeNeutralAngle(data):
    print('loaded before delete Images:', len(data))
    removeIndexlist = []
    for i in range(len(data['Steering'])):
        if data['Steering'][i] == 2:
            removeIndexlist.append(i)
    print('Removed Images:', len(removeIndexlist))
    data.drop(data.index[removeIndexlist], inplace=True)
    print('Remaining Images:', len(data))
    return data

def changeSteeringMatrix(path):
    columns = ['Center', 'Steering']
    data = pd.DataFrame()
    dataNew = pd.read_csv(os.path.join(path, 'log_0.csv'), names=columns)
    dataNew['Center'] = dataNew['Center'].apply(getName)
    dataNew['Steering']=dataNew['Steering'].apply(getSteeringInMatrix)
    data = data.append(dataNew, True)
    print(' ')
    print('Total Images Imported', data.shape[0])
    return data

def changeSteeringAngle(path):
    columns = ['Center', 'Steering']
    data = pd.DataFrame()
    dataNew = pd.read_csv(os.path.join(path, 'log_0.csv'), names=columns)
    dataNew['Center'] = dataNew['Center'].apply(getName)
    dataNew['Steering'] = dataNew['Steering'].apply(getSteeringInAngle)
    data = data.append(dataNew, True)
    print(' ')
    print('Total Images Imported', data.shape[0])
    return data

#### STEP 2 - VISUALIZE AND BALANCE DATA
def balanceData(data,display=True):
    nBin = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data['Steering'], nBin)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
        #data already balanced
    # removeindexList = []
    # for j in range(nBin):
    #     binDataList = []
    #     for i in range(len(data['Steering'])):
    #         if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
    #             binDataList.append(i)
    #     binDataList = shuffle(binDataList)
    #     binDataList = binDataList[samplesPerBin:]
    #     removeindexList.extend(binDataList)
    #
    # print('Removed Images:', len(removeindexList))
    # data.drop(data.index[removeindexList], inplace=True)
    # print('Remaining Images:', len(data))
    #
    #
    # if display:
    #     hist, _ = np.histogram(data['Steering'], (nBin))
    #     plt.bar(center, hist, width=0.06)
    #     plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
    #     plt.show()
    return data

#### STEP 3 - PREPARE FOR PROCESSING
def loadData(path, data):
  imagesPath = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append(f'{path}/IMG/{indexed_data[0]}')
    steering.append(indexed_data[1])
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  return imagesPath, steering


#### STEP 5 - AUGMENT DATA
def augmentImage(imgPath,steering):
    img =  mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

        steeringnp=np.array(steering)
        frwd=np.array([0,1,0])
        frwdl=np.array([1,0,0])
        frwdr=np.array([0,0,1])

        if (steeringnp==frwdl).all():
            steering=[0,0,1]
        elif (steeringnp==frwdr).all():
            steering=[1,0,0]
        elif (steeringnp==frwd).all():
            steering=steering

    return img, steering

# imgRe,st = augmentImage('DataCollected/IMG18/Image_1601839810289305.jpg',0)
# #mpimg.imsave('Result.jpg',imgRe)
# plt.imshow(imgRe)
# plt.show()

#### STEP 6 - PREPROCESS
def preProcess(img):
    img = img[54:120,:,:]
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# imgRe = preProcess(mpimg.imread('DataCollected/IMG18/Image_1601839810289305.jpg'))
# # mpimg.imsave('Result.jpg',imgRe)
# plt.imshow(imgRe)
# plt.show()

#### STEP 7 - CREATE MODEL

with tf.device('/device:GPU:0'):
    def createModel():
      model = Sequential()

      model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
      model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
      model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
      model.add(Convolution2D(64, (3, 3), activation='elu'))
      model.add(Convolution2D(64, (3, 3), activation='elu'))

      model.add(Flatten())
      model.add(Dense(100, activation = 'elu'))
      model.add(Dense(50, activation = 'elu'))
      model.add(Dense(10, activation = 'elu'))
      model.add(Dense(3,activation=softmax))

      model.compile(Adam(learning_rate=0.0001),loss='mse')
      return model

#### STEP 8 - TRAINNING
def dataGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcess(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))
