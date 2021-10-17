'''File contains the trainer class

Complete the functions train() which will train the network given the dataset and hyperparams, and the function __init__ to set your network topology for each dataset
'''
import numpy as np
import sys
import pickle

import nn

from util import *
from layers import *

class Trainer:
	def __init__(self,dataset_name):
		self.save_model = False
		if dataset_name == 'MNIST':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readMNIST()
			# Add your network topology along with other hyperparameters here
			#print(self.XTrain.shape[1])
			self.batch_size = 20
			self.epochs = 5#35
			self.lr = 0.02#0.06
			self.nn = nn.NeuralNetwork(10,self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1],20,'relu'))
			self.nn.addLayer(FullyConnectedLayer(20,10,'softmax')) #7(91.93),3(92.25),4(91.68),5(92.44),6(91.69),60(92.29),10(88.75),335(92.19)
			#7(93.3),3(92.91),4(93.23),5(93.03),6(93.49),60(92.97),10(93.29),335(93.15)


		if dataset_name == 'CIFAR10':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCIFAR10()
			self.XTrain = self.XTrain[0:5000,:,:,:]
			self.XVal = self.XVal[0:1000,:,:,:]
			self.XTest = self.XTest[0:1000,:,:,:]
			self.YVal = self.YVal[0:1000,:]
			self.YTest = self.YTest[0:1000,:]
			self.YTrain = self.YTrain[0:5000,:]

			self.save_model = True
			self.model_name = "model.p"

			# Add your network topology along with other hyperparameters here
			self.batch_size = 10
			self.epochs = 10
			self.lr = 0.15
			self.nn = nn.NeuralNetwork(10,self.lr)
			self.nn.addLayer(ConvolutionLayer([3, 32, 32], [3, 3], 10, 2, 'relu'))
			self.nn.addLayer(AvgPoolingLayer([10,15,15],[3,3],3))
			self.nn.addLayer(FlattenLayer())
			self.nn.addLayer(FullyConnectedLayer(250,10,'softmax'))#4,4 1(42.69),2(39.3),3(40.8),
			#self.nn.addLayer(FullyConnectedLayer(25,10,'softmax'))#3,3 1(41.2), 2(35.8), 3(43), 4(33.9),5(43.3),6(42.9),54(35.0),200(41.6),335(40.3),1234(38.8)

		if dataset_name == 'XOR':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readXOR()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 15
			self.epochs = 20
			self.lr = 0.25 #0.15
			self.nn = nn.NeuralNetwork(2,self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1], 4, 'softmax'))
			self.nn.addLayer(FullyConnectedLayer(4, 2, 'softmax')) #think it is minimal topology and worked good for multiple seeds(1,3,10,100,156,179,335,3345,3355)


		if dataset_name == 'circle':
			self.XTrain, self.YTrain, self.XVal, self.YVal, self.XTest, self.YTest = readCircle()
			# Add your network topology along with other hyperparameters here
			self.batch_size = 30 
			self.epochs = 50
			self.lr = 0.35
			self.nn = nn.NeuralNetwork(2,self.lr)
			self.nn.addLayer(FullyConnectedLayer(self.XTrain.shape[1], 3, 'softmax'))
			self.nn.addLayer(FullyConnectedLayer(3, 2, 'softmax')) #seed =1,10,100,334,256,2564,25643       
	def train(self, verbose=True):
		# Method for training the Neural Network
		# Input
		# trainX - A list of training input data to the neural network
		# trainY - Corresponding list of training data labels
		# validX - A list of validation input data to the neural network
		# validY - Corresponding list of validation data labels
		# printTrainStats - Print training loss and accuracy for each epoch
		# printValStats - Prints validation set accuracy after each epoch of training
		# saveModel - True -> Saves model in "modelName" file after each epoch of training
		# loadModel - True -> Loads model from "modelName" file before training
		# modelName - Name of the model from which the funtion loads and/or saves the neural net
		
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		for epoch in range(self.epochs):
			# A Training Epoch
			if verbose:
				print("Epoch: ", epoch)

			#TODO
			# if(epoch!=0 and epoch%10==0):
			# 	print('sleep')
			# 	time.sleep(20)
			# 	print('wake')
			my1=np.random.permutation(self.XTrain.shape[0])
			XTrain=self.XTrain[my1]
			YTrain=self.YTrain[my1]
			# Shuffle the training data for the current epoch


			# Initializing training loss and accuracy
			trainLoss = 0
			trainAcc = 0

			# Divide the training data into mini-batches
			numBatches=int(XTrain.shape[0]/self.batch_size)
			a=self.batch_size
			for i in range(numBatches):
				acts=self.nn.feedforward(XTrain[i*a:(i+1)*a])
				trainLoss+=self.nn.computeLoss(YTrain[i*a:(i+1)*a],acts)
				bc=np.copy(acts[-1])
				cd=np.max(bc,axis=1,keepdims=True)
				bc=(bc==cd)
				trainAcc+=self.nn.computeAccuracy(YTrain[i*a:(i+1)*a],bc)
				self.nn.backpropagate(acts,YTrain[i*a:(i+1)*a])

				# Calculate the activations after the feedforward pass
				# Compute the loss  
				# Calculate the training accuracy for the current batch
				# Backpropagation Pass to adjust weights and biases of the neural network

			# END TODO
			# Print Training loss and accuracy statistics
			trainAcc /= numBatches
			trainLoss /= numBatches
			#trainLoss /= numBatches
			if verbose:
				print("Epoch ", epoch, " Training Loss=", trainLoss, " Training Accuracy=", trainAcc)
			
			if self.save_model:
				model = []
				for l in self.nn.layers:
					# print(type(l).__name__)
					if type(l).__name__ != "AvgPoolingLayer" and type(l).__name__ != "FlattenLayer": 
						model.append(l.weights) 
						model.append(l.biases)
				pickle.dump(model,open(self.model_name,"wb"))
				print("Model Saved... ")

			# Estimate the prediction accuracy over validation data set
			if self.XVal is not None and self.YVal is not None and verbose:
				_, validAcc = self.nn.validate(self.XVal, self.YVal)
				print("Validation Set Accuracy: ", validAcc, "%")

		pred, acc = self.nn.validate(self.XTest, self.YTest)
		print('Test Accuracy ',acc)

