
import numpy as np
from numpy import linalg as la
import cPickle, gzip
import matplotlib.pyplot as plt
import copy

# INFO: 'mnist.pkl.gz' must be in the same directory
# as mnist.py for now in order for the data to load

globBias = True

# loads the mnist dataset from 'mnist.pkl.gz'
# into numpy arrays for training, validation
# and testing datasets
def load(oneHot=False, bias=True):
    global globBias

    # Load the mnist dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # PROCESS TRAINING DATA
    # get training points and labels
    trainx, trainy = train_set
    trainX = np.asarray(trainx)
    trainY = np.asarray(trainy)

    # get training set size
    [ntrain, dtrain] = trainX.shape

    # reshape trainY so that it doesnt have
    # an empty dimension
    trainY = np.reshape(trainY, (ntrain, 1))

    # PROCESS VALIDATION DATA
    # get validation points and labels
    validx, validy = train_set
    validX = np.asarray(validx)
    validY = np.asarray(validy)

    # get validation set size
    [nvalid, dvalid] = validX.shape

    # reshape validY so that it doesnt have
    # an empty dimension
    validY = np.reshape(validY, (nvalid, 1))

    # PROCESS TESTING DATA
    # get tesing points and labels
    testx, testy = test_set
    testX = np.asarray(testx)
    testY = np.asarray(testy)

    # get validation set size
    [ntest, dtest] = testX.shape

    # reshape testY so that it doesnt have
    # an empty dimension
    testY = np.reshape(testY, (ntest, 1))

    # by default, we make the points homegenious
    # with the bias added at the end
    if bias:
        trainX = np.append(trainX, np.ones((ntrain, 1)), axis=1)
        validX = np.append(validX, np.ones((nvalid, 1)), axis=1)
        testX = np.append(testX, np.ones((ntest, 1)), axis=1)
    else:
        globBias = False

    # if oneHot selected, convert all labels to oneHot
    if oneHot:
        trainVals = np.unique(trainY)
        trainY  = (trainVals == trainY)*1.0
        validVals = np.unique(validY)
        validY  = (validVals == validY)*1.0
        testVals = np.unique(testY)
        testY  = (testVals == testY)*1.0

    return trainX, trainY, validX, validY, testX, testY

def convertToBin(X):
    minX = np.min(X)
    maxX = np.max(X)
    thresh = (float(np.max(X)) - float(np.min(X)))/2.0
    convertX = 1*(X > thresh)
    return convertX

def convertToNegPos(X):
    minX = np.min(X)
    maxX = np.max(X)
    thresh = (float(np.max(X)) - float(np.min(X)))/2.0
    convertX = 2*(X > thresh) - 1
    return convertX

# shuffles a dataset of points, X and labels Y
def shuffle(X, Y):
    [nx, dx] = X.shape
    [ny, dy] = Y.shape

    # combine X and Y into one dataset
    D = np.append(X, Y, axis=1)

    # randomly shuffle the dataset
    D = np.random.permutation(D)

    # separate D back into X and Y
    shuffX = np.reshape(D[:, 0:dx], (nx, dx))
    shuffY = np.reshape(D[:, 0:dy], (ny, dy))

    return shuffX, shuffY


# TODO
# can be used for all types of vecs
def addGaussNoise(X):
    print("adding gaussing noise")


# randomly selects a bit to flip based on
# if a random number uniformly chosen between
# 0 and 100 is <= percent
# only used for [0, 1] vecs or [-1, 1] vecs
def addBitflipNoise(x, percent):
    # cast x to integer
    x = x.astype(int)

    d = x.size
    x = np.reshape(x, (1, d))

    if globBias:
        # temporalaly delete the bias
        x = np.delete(x, d-1, 1)
        d = d - 1

    # flip array will be 1 if randint int from 0-100
    # is below or equal to the noise percentage
    flip = 1*(np.random.randint(0, 100, (1, d)) <= percent)
    x = np.reshape(x, (1, d))
    vals = np.unique(x)
    if vals.size > 2:
        print("[Error]: not a valid data format for bitflip")
        if globBias:
            # add back the bias
            x = np.append(x, np.ones((1, 1)), axis=1)
        return x
    elif vals.size == 2:
        if all(i in vals for i in [-1, 1]):
            # temporarily convert to binary
            x = (x + 1)/2

            # flip any bits that coorespond with
            # the flip vector
            nx = x ^ flip

            # ensure that the noise vector will
            # have at leats 1 difference
            while(la.norm(nx - x) == 0):
                flip = 1*(np.random.randint(0, 100, (1, d)) <= percent)
                nx = x ^ flip

            # convert back to [-1, 1]
            nx = 2*(nx > 0.5) - 1

            if globBias:
                # add back the bias
                nx = np.append(nx, np.ones((1, 1)), axis=1)

            return nx

        elif all(i in vals for i in [0, 1]):
            # flip any bits that coorespond with
            # the flip vector
            nx = x ^ flip

            # ensure that the noise vector will
            # have at leats 1 difference
            while(la.norm(nx - x) == 0):
                flip = 1*(np.random.randint(0, 100, (1, d)) <= percent)
                nx = x ^ flip

            if globBias:
                # add back the bias
                nx = np.append(nx, np.ones((1, 1)), axis=1)

            return nx
        else:
            print("[Error]: not a valid data format for bitflip")
            if globBias:
                # add back the bias
                x = np.append(x, np.ones((1, 1)), axis=1)
            return x
    else:
        print("[Error]: not a valid data format for bitflip")
        if globBias:
            # add back the bias
            x = np.append(x, np.ones((1, 1)), axis=1)
        return x

def batch(X, Y, batchSize):
    [nx, dx] = X.shape
    [ny, dy] = Y.shape

    if (batchSize > nx):
        #print("[WARNING]: requested batch size greater than\
        #        sample size, changing batch size to sample size")
        batchSize = nx

    batchX = np.zeros((batchSize, dx))
    batchY = np.zeros((batchSize, dy))

    # grab batchSize random rows from D
    rows = []
    for n in range(batchSize):
        # find a random row we havent already used
        r = np.random.randint(0, nx)
        while(r in rows):
            r = np.random.randint(0, nx)
        rows.append(r)

        # copy this row to batchX and batchY
        batchX[n, :] = copy.deepcopy(X[r, :])
        batchY[n, :] = copy.deepcopy(Y[r, :])

    return batchX, batchY


#***   DEBUGGING AID   ***#

# plots a number. Will automatically reshape
# the number so it takes a (,d) row from the
# training, validation or testing dataset.
# This function blocks so it is only recommended
# to aid in debugging
def plotNum(figNum, x):
    d = x.size
    x = np.reshape(x, (1, d))

    plt.figure(figNum)
    if globBias:
        img = np.reshape(x[0, 0:d-1], (28, 28))
    else:
        img = np.reshape(x[0, 0:d], (28, 28))
    plt.imshow(img)
    #plt.show()
