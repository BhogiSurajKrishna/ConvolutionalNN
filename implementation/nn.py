import numpy as np
import random
from util import oneHotEncodeY
import itertools

class NeuralNetwork:

    def __init__(self, out_nodes, lr):
        # Method to initialize a Neural Network Object
        # Parameters
        # out_nodes - number of output nodes
        # alpha - learning rate
        # batchSize - Mini batch size
        # epochs - Number of epochs for training
        self.layers = []
        self.out_nodes = out_nodes
        self.alpha = lr

    def addLayer(self, layer):
        # Method to add layers to the Neural Network
        self.layers.append(layer)


    def computeLoss(self, Y, predictions):
        # Returns the crossentropy loss function given the prediction and the true labels Y
        # TODO 
        # print(Y.shape)
        # print(predictions[-1].shape)
        cons=10**(-8)
        A1=np.copy(predictions[-1])
        A1 = (A1>0)*A1+cons*(A1<=0)        
        a=np.sum(np.sum(Y*(np.log(A1))))# + (1-Y)*(np.log(B1)))
        return -a/Y.shape[0]
        raise NotImplementedError

        # END TODO
    def computeAccuracy(self, Y, predLabels):
        # Returns the accuracy given the true labels Y and predicted labels predLabels
        correct = 0
        for i in range(len(Y)):
            if np.array_equal(Y[i], predLabels[i]):
                correct += 1
        accuracy = (float(correct) / len(Y)) * 100
        return accuracy

    def validate(self, validX, validY):
        # Input 
        # validX : Validation Input Data
        # validY : Validation Labels
        # Returns the validation accuracy evaluated over the current neural network model
        valActivations = self.feedforward(validX)
        pred = np.argmax(valActivations[-1], axis=1)
        validPred = oneHotEncodeY(pred, self.out_nodes)
        validAcc = self.computeAccuracy(validY, validPred)
        return pred, validAcc

    def feedforward(self, X):
        # Input
        # X : Current Batch of Input Data as an nparray
        # Output
        # Returns the activations at each layer(starting from the first layer(input layer)) to 
        # the output layer of the network as a list of np multi-dimensional arrays
        # Note: Activations at the first layer(input layer) is X itself     
        # TODO
        activations1=[X]
        Y=np.copy(X)
        for lay in self.layers:
            Y=lay.forwardpass(Y)
            activations1.append(Y)
        return activations1
        raise NotImplementedError
        # END TODO

    def backpropagate(self, activations, Y):
        # Input
        # activations : The activations at each layer(starting from second layer(first hidden layer)) of the
        # neural network calulated in the feedforward pass
        # Y : True labels of the training data
        # This method adjusts the weights(self.layers's weights) and biases(self.layers's biases) as calculated from the
        # backpropagation algorithm
        # Hint: Start with derivative of cross entropy from the last layer

        # TODO
        L=len(activations)
        Lminus1=len(self.layers)
        cons=10**(-8)
        A1=np.copy(activations[-1])
        A1 = (A1>0)*A1+cons*(A1<=0)
        lthder=-1*(Y/A1)# - (1-Y)/B1)
        for i in range(Lminus1):
            j=Lminus1-i-1
            lthder=self.layers[j].backwardpass(self.alpha,activations[j],lthder)
        #raise NotImplementedError
        # END TODO