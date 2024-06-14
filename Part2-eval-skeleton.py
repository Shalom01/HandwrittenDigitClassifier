import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pudb
import numpy as np

# ============================================================
# define your network here, just copy the definition you made in the traing script
# ============================================================
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

# ============================================================
# just copy the definition you made in the traing script
# ============================================================
def compute_accuracy(prediction,gt_logits):
    acc = 0
    for i in range(0, len(prediction)):
        if np.argmax(prediction[i]) == gt_logits[i]:
            acc = acc + 1
    return acc /len(prediction)

# ==================================================================
# Build components
# ==================================================================
batch_size = 100

# build data transformation
transform = transforms.ToTensor()

# build dataset for validation split
dataset_val = datasets.MNIST('data', train=False, download=True,transform=transform)

# build dataloader
dataloader_val = torch.utils.data.DataLoader(dataset_val,batch_size=batch_size,shuffle=True,num_workers=4)

# build network
net = Net()


# build loss function, use cross entropy
loss_fn = torch.nn.CrossEntropyLoss()

# Load the checkpoint we made, and load the stat into the model
# take a look at:
# https://pytorch.org/docs/stable/generated/torch.load.html
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

checkpoint = torch.load('epoch-0003.pth') #loading the model produced following the 4th epoch
net.load_state_dict(checkpoint)

# ==================================================================
# Eval loop
# ==================================================================
if __name__ == '__main__':
    net.eval() # hopefully you remember why this is here
    cur_val_losses = []
    cur_val_accuracies = []
    for im, label in dataloader_val:
        with torch.no_grad():
                # Center data and pad to 32x32, for convienence
                im = im - 0.5
                im = torch.nn.functional.pad(im,[2,2,2,2,0,0,0,0])
                
                # get prediction from network
                pred = net(im)

                # compute loss
                loss = torch.nn.functional.cross_entropy(pred, label)

                # compute accuracy
                accuracy = compute_accuracy(pred, label)

                #add loss and accuracy to their lists
                cur_val_losses.append(loss.item())
                cur_val_accuracies.append(accuracy)

    avg_val_loss = np.array(cur_val_losses).mean()
    avg_val_accuracies = np.array(cur_val_accuracies).mean()
    print(f'Average validation loss: {avg_val_loss}')
    print(f'Average validation Accuracy: {avg_val_accuracies}')