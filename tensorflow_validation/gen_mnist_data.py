#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import mnist as mn

# uncomment for mnist
trainX, trainY, validX, validY, testX, testY = mn.load(oneHot=True, bias=False)
trainX = 255*trainX
trainX = trainX.astype(int)
trainY = trainY.astype(int)
validX = 255*validX
validX = validX.astype(int)
validY = validY.astype(int)
testX = 255*testX
testX = testX.astype(int)
testY = testY.astype(int)

[numTrain, dimTrain] = trainX.shape
[numTest, dimTest] = testX.shape

_,k = trainY.shape

# write contents to training data file
data = []
for i in range(numTrain):
    sample = trainX[i, :].tolist()
    for j in range(k):
        sample.append(trainY[i, j])
    data.append(sample)
file = open('mnist_train', 'wb')
for sample in data:
    file.write(bytes(sample))
file.close()

# write contents to testing data file
data = []
for i in range(numTest):
    sample = testX[i, :].tolist()
    for j in range(k):
        sample.append(testY[i, j])
    data.append(sample)
file = open('mnist_test', 'wb')
for sample in data:
    file.write(bytes(sample))
file.close()
