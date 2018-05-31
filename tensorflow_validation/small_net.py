#!/usr/bin/env python

"""
A two layer MLP implemented in Tensor Flow and trained on the MNIST dataset.
User is free to choose the number of neurons in each hidden layer, learning
rate, number of training iterations and whether or not to plot training results. 
"""

import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import mnist as mn
from timeit import default_timer as timer
import matplotlib.ticker as tkr

print "Tensorflow Version: " + tf.__version__ + "\n"

numArgs = 5;
printWeights = 0

if __name__ == '__main__':

    # retrieve command line args
    if (len(sys.argv) < numArgs + 1):
        print("[ERROR] not enough cmd line arguments")
        print("[USAGE] <Nh1> <Nh2> <eta> <numIter> <plots>")
        sys.exit()

    Nh1 = int(sys.argv[1])
    Nh2 = int(sys.argv[2])
    eta = float(sys.argv[3])
    numIter = int(sys.argv[4])
    plots = int(sys.argv[5])

    trainX, trainY, validX, validY, testX, testY = mn.load(oneHot=True, bias=False)

    trainX = np.round(255.0*trainX)/255.0
    validX = np.round(255.0*validX)/255.0
    testX = np.round(255.0*testX)/255.0

    ntrain, dtrain = trainX.shape
    _, k = trainY.shape
    ntest, dtest = testX.shape

    Ni = dtrain
    No = k

    # error list for plotting errors
    E = []

    print("Network dimensions: " + str([Ni, Nh1, Nh2, No]))
    print("Learning rate: " + str(eta))
    print("Number of iterations: " + str(numIter) + "\n")

    low, high = [-1, 1]

    weightsInit = {
        'h1': np.random.uniform(low, high, (Ni, Nh1)),
        'h2': np.random.uniform(low, high, (Nh1, Nh2)),
        'out': np.random.uniform(low, high, (Nh2, No))
    }

    biasInit = {
        'h1': np.random.uniform(low, high, (Nh1)),
        'h2': np.random.uniform(low, high, (Nh2)),
        'out': np.random.uniform(low, high, (No))
    }

    #                      TENSORFLOW GRAPH COMPONENTS                         #
    # ************************************************************************ #
    # graph input
    inputs = tf.placeholder(tf.float32, [None, Ni])
    targets = tf.placeholder(tf.float32, [None, No])

    # initialize layer weights
    W = {
        'h1': tf.Variable(weightsInit['h1'], dtype=np.float32),
        'h2': tf.Variable(weightsInit['h2'], dtype=np.float32),
        'out': tf.Variable(weightsInit['out'], dtype=np.float32)
    }

    # initialize layer biases
    B = {
        'h1': tf.Variable(biasInit['h1'], dtype=np.float32),
        'h2': tf.Variable(biasInit['h2'], dtype=np.float32),
        'out': tf.Variable(biasInit['out'], dtype=np.float32)
    }

    # output and hidden state of network when feeding in inputs
    hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs, W['h1']), B['h1']))
    hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, W['h2']), B['h2']))
    outputs = tf.nn.sigmoid(tf.add(tf.matmul(hidden2, W['out']), B['out']))

    loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(outputs, targets), 2))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(loss)

    correct = tf.cast(tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), tf.float32)

    accuracy = tf.reduce_mean(correct)

    init = tf.global_variables_initializer()

    # ************************************************************************ #

    EA = []
    expavg = 0
    alpha = 0.04

    #                          TENSORFLOW SESSION                              #
    # ************************************************************************ #
    with tf.Session() as sess:
        sess.run(init)

        startTime = timer()

        # train the network
        index = 0
        for i in range(numIter):

            if (index == ntrain):
                index = 0

            # grab a randomly chosen batch from the training data
            sample = np.reshape(trainX[index, :], (1, Ni))
            label = np.reshape(trainY[index, :], (1, No))

            # run a step of gradient descent and get the loss
            _, err = sess.run([optimizer, loss], feed_dict={inputs: sample, targets: label})

            E.append(err)

            expavg = (alpha * err) + (1.0 - alpha) * expavg
            EA.append(expavg)

            index += 1

            # report training progress
            progress = int(float(i)/float(numIter)*100.0)
            sys.stdout.write('\rTraining: ' + str(progress) + '%')
            sys.stdout.flush()

        sys.stdout.write('\rTraining: 100%\n\n')
        sys.stdout.flush()

        trainTime = timer() - startTime
        print("Total training time: %.02fs\n" % (trainTime))

        correctRate = sess.run(accuracy, feed_dict={inputs: testX, targets: testY})

        print("Accuracy: " + str(format(100.0*correctRate, '0.1f')) + "%")

        TW = [W['h1'].eval(), W['out'].eval()]
        TB = [B['h1'].eval(), B['out'].eval()]

        sess.close()

    # ************************************************************************ #

    # collect the information from the mpi MLP
    MPI_EA = []
    expavg = 0
    mpi_err = np.loadtxt("training_error.txt")
    for i in range(mpi_err.shape[0]):
        expavg = (alpha * mpi_err[i]) + (1.0 - alpha) * expavg
        MPI_EA.append(expavg)


    if printWeights:
        print("\nInitial Weights")
        print weightsInit['h1']
        print weightsInit['out']
        print biasInit['h1']
        print biasInit['out']

        print("\nFinal Weights")
        print "Wh1", TW[0]
        print "Wout", TW[1]
        print "Bh1", TB[0]
        print "Bout",TB[1]

    # plot error vs iteration
    if plots:
        fmt = tkr.ScalarFormatter()
        fmt.set_powerlimits((-3, 3))
        f, axarr = plt.subplots(1, 2)
        plt.suptitle("Training Progress", fontsize=20)
        axarr[0].plot(range(len(EA)), EA)
        axarr[0].set_xlim([0, len(EA)])
        axarr[0].set_ylim([0, max(EA)])
        axarr[0].xaxis.set_major_formatter(fmt)
        axarr[0].set_title('Tensor Flow')
        axarr[0].set_xlabel('Iteration')
        axarr[0].set_ylabel('Average Squared Error')
        axarr[1].plot(range(len(MPI_EA)), MPI_EA)
        axarr[1].set_xlim([0, len(MPI_EA)])
        axarr[1].set_ylim([0, max(EA)])
        axarr[1].xaxis.set_major_formatter(fmt)
        axarr[1].set_title('MPI')
        axarr[1].set_xlabel('Iteration')
        plt.show()
