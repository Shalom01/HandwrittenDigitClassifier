# ========================================================================================
# A checker that compares analytical gradients with numerically computed ones.
# This method is based on finite differences.
# See this for a general understanding: https://en.wikipedia.org/wiki/Finite_difference
#
# This script tests >>>>>Softmax_cross_entropy_loss<<<<<<
# You can also use this script to test other layers you implement but you may also need to
# to perform modifications to compute gradients for tensors other than the inputs
# or if the layer has additional inputs.
# ========================================================================================

import numpy as np

# ========================================================================================
# This is a base class for layers.
# It includes methods used for backprop and optimization.
# ========================================================================================
class Layer:
    '''
    Base class for all the layers we will define
    '''
    def __init__(self):
        '''
        Setup layer parameters, and gradients
        '''
        raise NotImplemented

    def forward(self,x):
        '''
        Computes forward pass, also returnes a context variable to store anything needed for backward pass
        Note: You will need to choose what to store in the ctx variable
        '''
        raise NotImplemented

    def backward(self,ctx,output_grad):
        '''
        Computes gradients for input tensor and any parameters, using the gradient of the output tensor, and the context saved from forward
        '''
        raise NotImplemented

    def zero_grads(self):
        '''
        Set all gradients to 0, does nothing if this layer has no parameters
        '''
        pass # default, do nothing

    def sgd_step(self,lr):
        '''
        Take a step of SGD used the stored gradients, does nothing if this layer has no parameters
        '''
        pass # default, do nothing


# ========================================================================================
# Define your layer here.
# You can copy it from your implementation.
# ========================================================================================
class Softmax_cross_entropy_loss(Layer):
    def __init__(self,n_logit_classes):
        self.n_classes = n_logit_classes #nunber of output nodes of Softmax

    def forward(self,x,logits):
        ctx = {'inputs': x, 'logits' : logits}

        #compute softmax probabilities for each sample
        input_maxes = np.reshape(np.max(x, axis = 1), (len(x), 1)) #the maximum input of each sample
        x = np.subtract(x, input_maxes) #subtract the maxes from the inputs 
        softmax_probs = np.exp(x)/np.reshape(np.sum(np.exp(x), axis=1), (len(x), 1)) #the softmax probabilities

        #calculate the actual probabilites for each sample
        actual_probs = np.eye(self.n_classes)[logits.squeeze()]

        #calculate cross entropy loss
        cross_entropy = -1*np.multiply(actual_probs, np.log(softmax_probs))
        
        #average the loss across batches
        cross_entropy_across_batches = np.sum(cross_entropy)/len(logits)
        return cross_entropy_across_batches, ctx

    def backward(self,ctx):
        '''
        Note: We assume this is the last layer, so it doesn't take an output_grad
        '''
        x = ctx['inputs'] #the input matrix
        labels = ctx['logits'] #the label vector

        #calculate softmax probabilities for each sample
        input_maxes = np.reshape(np.max(x, axis = 1), (len(x), 1)) #the maximum input of each sample
        x = np.subtract(x, input_maxes) #subtract the maxes from the inputs 
        softmax_probs = np.exp(x)/np.reshape(np.sum(np.exp(x), axis=1), (len(x), 1))

        #calculate actual probabilities for each sample
        actual_probs = np.eye(self.n_classes)[labels.squeeze()]

        #the input gradient to be returned to the linear layers
        input_grad = softmax_probs - actual_probs
        return input_grad

# create the layer here
layer = Softmax_cross_entropy_loss(10)

# helpers
x1_shape = (1, 10)
epsilon = 0.000001

# randomly create a tensor
x1 = np.random.rand(x1_shape[0], x1_shape[1])
labels = np.random.randint(low=0, high=10, size=(1, len(x1)))

# compute analytical grad using the backward pass you implemented in the layer
# you might need more inputs than x1
y1, y1_ctx = layer.forward(x1, labels)

print(y1)
print(y1_ctx)

x1_analytical_grad = layer.backward(y1_ctx)

# ========================================================================================
# Computes the numerical gradient with respect to the output sum.
# Loops though all the elements of the input tensor
# that perturb each element seperately.
# ========================================================================================

x1_numerical_grad = np.zeros(x1_shape)

for i in range(0, x1.shape[1]):
    # do another forward pass but modify the original x1 input by epsilon for the current element
    x2 = np.zeros((1, x1.shape[1]))
    x2[0, i] = epsilon
    y2, y2_ctx = layer.forward(x1 + x2, labels)
    print(y2)
 
    # compute numerical grad
    x1_numerical_grad[0, i] = (y2 - y1)/epsilon
    
# compare the numerical gradient with the analytical gradient
# you can use np.isclose()

print(x1_analytical_grad)
print(x1_numerical_grad)
print(np.isclose(x1_numerical_grad, x1_analytical_grad))
