'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        a=X@(self.weights) + self.biases
        if self.activation == 'relu':
            self.data=relu_of_X(a)
            return self.data
            raise NotImplementedError
        elif self.activation == 'softmax':
            self.data=softmax_of_X(a)
            return self.data
            raise NotImplementedError

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO      
    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        # compute gradient wrt trainable parameters
        # compute gradient wrt input
        # update parameters
        # return gradient wrt input
        a3=len(activation_prev)
        if self.activation == 'relu':
            dsigBydsum=gradient_relu_of_X(self.data,delta)
            delW = (activation_prev.T)@dsigBydsum
            delB = np.sum(dsigBydsum,axis=0,keepdims=True)
            delEBydelsig=dsigBydsum@((self.weights).T)
            self.weights-=lr*(delW/a3)
            self.biases-=lr*(delB/a3)
            return delEBydelsig
            #raise NotImplementedError
        elif self.activation == 'softmax':
            dsigBydsum=gradient_softmax_of_X(self.data,delta)
            delW = (activation_prev.T)@dsigBydsum
            delB = np.sum(dsigBydsum,axis=0,keepdims=True)
            delEBydelsig=dsigBydsum@((self.weights).T)
            self.weights-=lr*(delW/a3)
            self.biases-=lr*(delB/a3)
            return delEBydelsig
            #raise NotImplementedError

        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        pass
        # END TODO
class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        #a=np.convolve(X,self.weights,'valid')
        n,d,r,c=X.shape
        outputs=np.zeros([n,self.out_depth,self.out_row,self.out_col])
        s=self.stride
        for i in range(self.out_row):
            i1=min(r,i*s+self.filter_row)
            for j in range(self.out_col):
                j1=min(c,j*s+self.filter_col)
                outputs[:,:,i,j]=np.sum(X[:,np.newaxis,:,i*s:i1,j*s:j1]*(self.weights[np.newaxis,:,:,:,:]),axis=(2,3,4))
        outputs+=self.biases[np.newaxis,:,np.newaxis,np.newaxis]
        if self.activation == 'relu':
            self.data=relu_of_X(outputs)
            return self.data
            raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        ###############################################
        # END TODO
    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################
        s=self.stride
        n,d,r,c=activation_prev.shape
        outputs=np.zeros(activation_prev.shape)
        delW=np.zeros([self.out_depth,self.in_depth,self.filter_row,self.filter_col])
        delB=np.zeros(self.out_depth)
        if self.activation == 'relu':
            inp_delta = gradient_relu_of_X(self.data, delta)
            for i in range(self.out_row):
                i1=min(r,i*s+self.filter_row)
                for j in range(self.out_col):
                    j1=min(c,j*s+self.filter_col)
                    outputs[:,:,i*s:i1,j*s:j1]+=np.sum((self.weights[np.newaxis,:,:,:,:])*inp_delta[:,:,np.newaxis,i:i+1,j:j+1],axis=1)
                    delW+=np.sum(activation_prev[:,np.newaxis,:,i*s:i1,j*s:j1]*inp_delta[:,:,np.newaxis,i:i+1,j:j+1],axis=0)
            delB=np.sum(inp_delta,axis=(0,2,3))
            self.biases-=lr*(delB/activation_prev.shape[0])
            self.weights-=lr*(delW/activation_prev.shape[0])
            return outputs
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        n,d,r,c=X.shape
        denom=self.filter_row*self.filter_col
        outputs=np.zeros([n,d,self.out_row,self.out_col])
        s=self.stride
        for i in range(self.out_row):
            i1=min(i*s+self.filter_row,r)
            for j in range(self.out_col):
                j1=min(j*s+self.filter_col,c)
                outputs[:,:,i,j]=np.sum(X[:,:,i*s:i1,j*s:j1],(2,3))/denom
        return outputs
        pass
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        derivs=np.repeat(np.repeat(delta,self.stride,axis=3),self.stride,axis=2)
        return derivs/(self.stride**2)
        pass
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)


    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        n,d,r,c=X.shape
        outputs=np.zeros([n,d,self.out_row,self.out_col])
        s=self.stride
        for i in range(self.out_row):
            i1=min(i*s+self.filter_row,r)
            for j in range(self.out_col):
                j1=min(j*s+self.filter_col,c)
                outputs[:,:,i,j]=np.max(X[:,:,i*s:i1,j*s:j1],(2,3))
        return outputs
        pass
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        n,d,r,c=activation_prev.shape
        o1=np.zeros(delta.shape)
        s=self.stride
        for i in range(self.out_row):
            i1=min(i*s+self.filter_row,r)
            for j in range(self.out_col):
                j1=min(j*s+self.filter_col,c)
                o1[:,:,i,j]=np.max(activation_prev[:,:,i*s:i1,j*s:j1],(2,3))
        intermed=np.repeat(np.repeat(o1,self.stride,axis=3),self.stride,axis=2)
        deltaexp=np.repeat(np.repeat(delta,self.stride,axis=3),self.stride,axis=2)
        return (intermed==activation_prev)*deltaexp
        pass
        # END TODO
        ###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        self.a,self.b,self.c,self.d=0,0,0,0
        pass
    
    def forwardpass(self, X):
        # TODO
        self.a,self.b,self.c,self.d = X.shape
        return X.reshape(self.a,self.b*self.c*self.d)
        # print(X.shape)
        pass
    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.a,self.b,self.c,self.d)
        pass
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    return (X>0)*X
    #raise NotImplementedError
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    return (X>0)*delta
    #raise NotImplementedError
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    y=np.exp(X)
    outp=np.sum(y,axis=1,keepdims=True)
    return (y/outp)
    #raise NotImplementedError
    # END TODO  
def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    #Y=softmax_of_X(X)
    n=X.shape[0]
    out_nodes1=X.shape[1]
    result=np.zeros([n,out_nodes1])
    for i in range(out_nodes1):
        Z=np.zeros([n,out_nodes1])
        Z[:,i]=X[:,i]
        Y=delta*(X*(X[:,i].reshape(n,1)) - Z)
        result[:,i]=np.sum(-1*Y,1)
    return result
    #raise NotImplementedError
    # END TODO
