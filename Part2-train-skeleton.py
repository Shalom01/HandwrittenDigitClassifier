import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np

# ==================================================================
# Contruct our network
# We inherit from torch.nn.Module, this allows it to keep track of parameters of member variables
# There are multiple ways to add layers to your network:
#   1) each layer is it's down member variable
#   2) each layer is contained in a torch.nn.Sequential
#   3) each layer is contained in a torch.nn.ModuleList
# Note: Do NOT place layers in a list, doing so will prevent their parameters from being found
# Note: for torch.nn.Modules, you can call forward() via the __call__() override
# IMPORTANT: For this model I want you to implement using: Conv2d, MaxPool2d, and ReLU, maybe Linear
#
# Take a look at:
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
# https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
# https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
# https://pytorch.org/docs/stable/generated/torch.reshape.html
# https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
# ==================================================================
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__() #TO DO: need to initialize parameters
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, dilation=1, padding=0) 
        self.conv_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, dilation=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=0, dilation=1)
        self.linear_1 = torch.nn.Linear(64*20*20, 128)
        self.linear_2 = torch.nn.Linear(128, 10)
        self.soft_max = torch.nn.Softmax(dim=1)


    def forward(self,x):
        '''
        N - batch size
        C - n image channels (1 for mnist)
        H - height of image, will be 32 for this skeleton code
        W - width of image, will be 32 for this skeleton code
        Input dimensions: [N,C,H,W]
        Output dimensions: [N,n_classes]
        Note: this will be called when you do net(x)
        '''
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(-1, 64*20*20) #flattening the input
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.soft_max(x)
        return x
    
# ==================================================================
# Copy your compute accuracy function from Part1 here
# I want you to use that because it uses Numpy, we will learn to convert torch->numpy tensors
# IMPORTANT: Numpy only, I DON'T WANT TO SEE PYTORCH HERE
# ==================================================================
def compute_accuracy(prediction,gt_logits):
    acc = 0
    for i in range(0, len(prediction)):
        if np.argmax(prediction[i]) == gt_logits[i]:
            acc = acc + 1
    return acc /len(prediction)

# ==================================================================
# Build components
# ==================================================================
# Magic variables
batch_size = 100

# this transformation is applied at to the output of datasets
# dataset provided images in PIL format, this transform also goes from [0,255] to [0,1]
transform=transforms.ToTensor()

# training and validation datasets
dataset_train = datasets.MNIST('data', train=True, download=True,transform=transform)
dataset_val = datasets.MNIST('data', train=False, download=True,transform=transform)

# dataloaders for datasets
# These will handling batching, shuffling of your data
# num_workers > 0 allows the code to prefetch data concurrently while training is occuring, increases training speed
dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=4)
dataloader_val = torch.utils.data.DataLoader(dataset_val,batch_size=batch_size,shuffle=True,num_workers=4)

# build network
net = Net()

# build loss function, use a CrossEntropyLoss
# take a look at: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
loss_fn = torch.nn.CrossEntropyLoss() #TO DO: set params

# build optimizer, and set learning rate
# unlike Part1 we will have an optimizer object manage parameter updates
# take a look at: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
learning_rate = 0.05
optimizer = torch.optim.SGD(net.parameters(), learning_rate) 

if __name__ == '__main__':
    # ==================================================================
    # Train loop
    # ==================================================================
    global_it = 0 # keep track of total number of iterations elapsed
    train_losses = [] # this stuff here is to track training statistics
    train_accuracies = []
    val_iterations = []
    val_losses = []
    val_accuracies = []
    for epoch in range(4): # an epoch is # of iterations to go through the whole training set
        net.train() # good practice, some networks have training dependent behaviour
        for im, label in dataloader_train: # the dataloader is an iterator, will shuffle, then go through all data once,
            # Center data and pad to 32x32, for convienence
            im = im - 0.5
            im = torch.nn.functional.pad(im,[2,2,2,2,0,0,0,0])

            # Zero out the gradients, just like in Part 1
            # take a look at: https://pytorch.org/docs/stable/optim.html
            optimizer.zero_grad()

            # get prediction from network, calling the object like a function invokes its forward() method
            pred = net(im)

            # compute the loss
            loss = torch.nn.functional.cross_entropy(pred, label) #SHALOM: I am using cross-entropy loss here

            # compute the accuracy, we will have to convert to numpy tensorfs first
            # .cpu() moves the tensor to the device, here it is pointless because it is already cpu, but must be done when we use gpus
            # .detach() tells torch that gradients will not flow through this operation, MUST be done before we get the numpy array
            # .numpy() gets the numpy array
            accuracy = compute_accuracy(pred.cpu().detach().numpy(),label.cpu().detach().numpy())

            # compute gradient in respect to the loss. Take a look here: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
            loss.backward()

            # take a step of SGD, call this through the optimizer!
            # take a look here: https://pytorch.org/docs/stable/optim.html
            optimizer.step()

            # store stats for this iteration
            # note, if the tensor is a scalar, .item() retrieves the value as a python primative
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)

            # print out stats every 10 its
            if global_it % 10 == 0:
                print(f'Iteration: {global_it} | Loss: {loss.item()} | Accuracy: {accuracy}')

            # increment
            global_it += 1

        print('Validating...')
        net.eval() # good practice, some networks have evaluation dependent behaviour
        cur_val_losses = []
        cur_val_accuracies = []
        for im, label in dataloader_val:
            # we don't need gradients so we wrap with this to not compute them
            # NOTE: much of this section looks like the training section!
            with torch.no_grad():
                # Center data and pad to 32x32, for convienence
                im = im - 0.5
                im = torch.nn.functional.pad(im,[2,2,2,2,0,0,0,0])

                # get prediction from network
                pred = net(im)

                # compute loss
                loss = torch.nn.functional.cross_entropy(pred, label) #SHALOM: I am using cross-entropy loss here

                # compute accuracy
                accuracy = compute_accuracy(pred, label)

                # collect stats for this iteration
                cur_val_losses.append(loss.item())
                # cur_val_accuracies.append(accuracy.item())
                cur_val_accuracies.append(accuracy)

        # take mean over entire validation set
        avg_val_loss = np.array(cur_val_losses).mean()
        avg_val_accuracies = np.array(cur_val_accuracies).mean()
        val_iterations.append(global_it)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracies)
        print(f'Average validation loss: {avg_val_loss}')
        print('Saving checkpoint')

        # save parameters of network
        # Note other things can have states (certain optimizers)
        # you would need to save these things to resume training
        # but we're not doing that here
        torch.save(net.state_dict(),f'epoch-{epoch:04d}.pth')

    print('Training complete')
    print('Plotting training stats')

    plt.plot(train_losses,label='Training')
    plt.plot(val_iterations,val_losses,label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Part2-training-loss.png')
    plt.clf()

    plt.plot(train_accuracies,label='Training')
    plt.plot(val_iterations,val_accuracies,label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Part2-training-acc.png')