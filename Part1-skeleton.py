
import math
from torchvision import datasets
import numpy as np
from PIL import Image
import pudb
import random
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

# ========================================================================================
# We provide you with this utility to shuffle and batch your data for you
# ========================================================================================
class Simple_batcher:
    '''
    A simple loader that shuffles data, and retrieves batches
    This is provided to you since the data input pipeline is not the focus of this part
    '''
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.init_shuffling()

    def init_shuffling(self):
        '''
        Generates a shuffled order of the data idxs, resets the index for that shuffled list
        '''
        self.shuffled_idxs = list(range(len(dataset)))
        random.shuffle(self.shuffled_idxs)
        self.current_position = 0

    def get_next_pos(self):
        '''
        Grabs the next index from the shuffled list, when we reach the end, shuffle again
        '''
        pos = self.current_position
        self.current_position += 1
        if self.current_position >= len(self.dataset):
            self.init_shuffling()
        return pos

    def get_batch(self):
        '''
        Gathers multiple images and labels (as logits) into a batch for training
        MNIST images are single channel 28x28, uint8 imagesq
        This dataloader will flatten each image into a 28*28 vector, and convert it into a float32

        N - Batch size
        Returned image shape: [N,784], dtype=float32
        Returned label shape: [N,1], dtype=int32
        '''
        batch_size = self.batch_size
        out_ims = []
        out_logits = []
        for n in range(batch_size):
            example = self.dataset[self.get_next_pos()]
            im = np.asarray(example[0]).flat
            label = example[1]
            out_ims.append(im)
            out_logits.append(label)
        out_ims = np.stack(out_ims,axis=0).astype(np.float32)
        out_logits = np.array([out_logits]).transpose(1,0)
        return out_ims, out_logits

# ========================================================================================
# This is a base class to base your layers on, this includes methods used for backprop and optimization
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
        Set all gradients to 0, d oes nothing if this layer has no parameters
        '''
        pass # default, do nothing

    def sgd_step(self,lr):
        '''
        Take a step of SGD used the stored gradients, does nothing if this layer has no parameters
        '''
        pass # default, do nothing

# ========================================================================================
# A simple Linear layer, should include a weight matrix and biases
# Input dimensions: [N,in_size]
# Output dimensions: [N,out_size]
#
# Implementation instructions:
# Initialize weights with a uniform distribution [-1/sqrt(in_size),1/sqrt(in_size)]
# Initialize biases to 0
# Layer should also contain variables for the gradients for the parameters
# Refer to the comments in the base class to see what needs to be implemented
# ========================================================================================
class Linear_layer(Layer): #applies the linear transformation

    def __init__(self,in_size,out_size):
        #self.out = np.zeros(out_size) #matrix of outputs - initially all zero
        self.weights = np.random.uniform(-1/math.sqrt(in_size), 1/math.sqrt(in_size), [in_size, out_size]) #init the weight vectors [W1, W2, ..., W_{out_size}]
        self.gradients = np.zeros([in_size, out_size]) #the gradients of each weight vector
        self.biases = np.zeros([out_size]) #init the biases for each weight vector

    def forward(self,x): #x is the input vector
        ctx = {'inputs': x} #context variable - should store the context for the backprop calculation
        output = np.dot(x, self.weights) + self.biases #compute linear transformation: [N, in_size] * [in_size, out_size]  -> [N, out_size] + [out_size]
        return output, ctx

    def backward(self,ctx,output_grad): #given output_grad (from following layer), and ctx variable (from preceeding layer)
        input_grad = np.dot(output_grad, self.weights.T)
        self.gradients = np.dot(ctx['inputs'].T, output_grad)/len(self.gradients)
        return input_grad

    def zero_grads(self): #set the gradients of the layer to 0
        self.gradients = np.zeros([len(self.gradients), len(self.gradients[0])])

    def sgd_step(self,lr): #lr = learning rate; perform an SGD step (update each W based on gradient)
        self.weights = self.weights - lr*self.gradients
        gradient_sums = np.sum(self.gradients, axis=0)/len(self.gradients)
        self.biases = self.biases - lr*gradient_sums

# ========================================================================================
# A simple RELU layer, output dimensions same as input dimensions
#
# Implementation instructions:
# Refer to the comments in the base class to see what needs to be implemented
# ========================================================================================
class RELU_layer(Layer):
    def __init__(self):
        pass

    def forward(self,x): #x is the input vector
        ctx = {'inputs': x}
        output = np.maximum(0, x)
        return output, ctx

    def backward(self,ctx,output_grad):
        x = ctx['inputs'] #retrieve the input vector from the ctx
        dRELU = np.maximum(x, np.zeros(x.shape))
        np.copyto(dRELU, np.ones(dRELU.shape), where= dRELU != 0)
        input_grad = np.multiply(dRELU, output_grad)
        return input_grad

# ========================================================================================
# A Softmax coss entropy loss layer
# This is a combination of a softmax layer and a cross entropy layer
# In theory you can implement these as two layers, but we will implement them as one for numerical reasons
# HINT: Try and work out the forward and backward by hand, see if any terms cancel out
# HINT: Try and not compute log(exp(x)), this can lead to NaNs
# HINT: at the end, you should average the loss across all batches
# Input dimensions: [N,n_classes]
# Output dimensions: [] - this is a scalar
#
# Implementation instructions:
# Refer to the comments in the base class to see what needs to be implemented
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

# ========================================================================================
# The class for the entire network
#
# Implementation instructions:
# Similar to the layer base class definition, see comments on individual methods
# ========================================================================================
class Net:
    def __init__(self):
        '''
        Construct all layers, and a loss function
        '''
        self.linear_1 = Linear_layer(784, 392) #[in, out]
        self.RELU_1 = RELU_layer()
        self.linear_2 = Linear_layer(392, 10)
        self.soft_max = Softmax_cross_entropy_loss(10)

    def forward(self,x):
        '''
        Pass the outputs of one layer, into the next, excluding the loss function
        Remember, each layer produces gradient context data that is needed for the backward pass
        '''

        grad_ctxs = [] #a list of gradients

        x, ctx = self.linear_1.forward(x)
        grad_ctxs.append(ctx)
        x, ctx = self.RELU_1.forward(x)
        grad_ctxs.append(ctx)

        x, ctx = self.linear_2.forward(x)
        grad_ctxs.append(ctx)

        return x, grad_ctxs

    def forward_with_loss(self,x,labels):
        '''
        Passes the prediction from the network to the loss function
        We just do this seperately because the loss layer is a little different from the rest
        '''
        pred, grad_ctxs = self.forward(x)
        loss, ctx = self.soft_max.forward(pred, labels)
        grad_ctxs.append(ctx)
        return pred, loss, grad_ctxs

    def compute_grads(self,grad_ctx):
        '''
        Do backprop, any gradients for the parameters should end up stored in the gradient variables in those layers
        '''
        output_grad = self.soft_max.backward(grad_ctx[3])
        output_grad = self.linear_2.backward(grad_ctx[2], output_grad)
        output_grad = self.RELU_1.backward(grad_ctx[1], output_grad)
        output_grad = self.linear_1.backward(grad_ctx[0], output_grad)

    def zero_grads(self):
        '''
        Set all stored gradients for all layers to 0
        '''
        self.linear_1.zero_grads()
        self.linear_2.zero_grads()

    def sgd_step(self,lr):
        '''
        Does a step of SGD for all layers 
        '''
        #i.e., updates the weights on each layer based on gradient (we should save the gradient at each layer)
        self.linear_1.sgd_step(lr)
        self.linear_2.sgd_step(lr)

# ========================================================================================
# Computes the accuracy of the prediction
# Prediction dimensions: [N, n_classes]
# Logit dimensions: [N, 1]
# The predicted class is determined as the highest value in the prediction
# ========================================================================================
def compute_accuracy(prediction,gt_logits):
    acc = 0
    for i in range(0, len(prediction)):
        if np.argmax(prediction[i]) == gt_logits[i][0]:
                acc = acc + 1
    return acc /len(prediction)

# ========================================================================================
# Setup for the components we will use
# We will use the MNIST training dataset provided by torch
# ========================================================================================
batch_size = 100
learning_rate = 0.05
dataset = datasets.MNIST('data', train=True, download=True)
loader = Simple_batcher(dataset,batch_size)
net = Net()
losses = [] # store losses throughout training
accuracies = [] # store accuracies throughout training

# ========================================================================================
# Training loop, provided to you
# ========================================================================================
for it in range(1000):
    net.zero_grads() # zero out the gradients
    ims, logits = loader.get_batch() # get the data
    ims = ims/127.5 - 1.0 # preprocessing, we just make the data [-1,1]
    pred, loss, grad_ctx = net.forward_with_loss(ims,logits) # do the forward pass
    accuracy = compute_accuracy(pred,logits) # compute accuracy
    losses.append(loss) # store loss
    accuracies.append(accuracy) # store accuracy

    net.compute_grads(grad_ctx) # backward pass
    net.sgd_step(learning_rate) # take a step of SGD

    if it % 10 == 0: # print the loss every 10 iterations
        print(loss)

# ========================================================================================
# Post training stuff, plot the stats from training
# The non-notebook version will save the images
# ========================================================================================
print('Training complete, plotting')

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('Part1-training-loss.png')
plt.clf()

plt.plot(accuracies)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.savefig('Part1-training-accuracy.png')
plt.clf()