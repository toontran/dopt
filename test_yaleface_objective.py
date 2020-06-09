import numpy as np
from random import shuffle
import copy
import tempfile
from argparse import Namespace

from functools import partial
from multiprocessing.pool import ThreadPool

# from keras.datasets import mnist
from sklearn.model_selection import KFold
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# torch
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch.nn import Module

# Maybe test the effect of higher dimensional space on BO
# Set up contraints?
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.expected_input_shape = (1, 1, 192, 168)
        self.conv1 = nn.Conv2d(1, args.conv1, args.conv1_kernel, 1) # padding is 0 by default, so 
                                            # we lose a pixel on each side.
        self.conv2 = nn.Conv2d(args.conv1, args.conv2, args.conv2_kernel, 1)
        self.dropout1 = nn.Dropout2d(args.dropout1)
        self.maxpool1 = nn.MaxPool2d(args.maxpool1)
        self.maxpool2 = nn.MaxPool2d(args.maxpool2)
        
        # Calculate the input of Linear layer
        conv1_out = self.get_output_shape(self.maxpool1, self.get_output_shape(self.conv1, self.expected_input_shape))
        conv2_out = self.get_output_shape(self.maxpool2, self.get_output_shape(self.conv2, conv1_out)) 
        fc1_in = np.prod(list(conv2_out)) # Flatten

        self.fc1 = nn.Linear(fc1_in, 38)

    def forward(self, x):
        x = self.conv1(x) # input: 1x192x168, output: 32x190x166
        x = F.relu(x)
        x = self.maxpool1(x) 
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.maxpool2(x) 
        x = self.dropout1(x) 
        x = torch.flatten(x, 1) 
        x = self.fc1(x) 
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        # output = torch.sigmoid(x)
        return output
    
    # TODO: Use from utils instead
    def get_output_shape(self, model, image_dim):
        return model(torch.rand(*(image_dim))).data.shape

def train(args, model, device, train_loader, optimizer, epoch):    
    model.train()
    loss_sum = 0
    count = 1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        # loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    
    loss_avg = loss_sum/len(train_loader.dataset)
#     print('\nEpoch: {}. Train set: Average loss: {:.4f}'.\
#           format(epoch, loss_avg))
    
    return loss_avg
    
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    
    return test_loss, test_acc

def run_train_net_once(yaleData, train_idx, test_idx, args):
    trainFaces = torch.utils.data.Subset(yaleData, train_idx)
    testFaces = torch.utils.data.Subset(yaleData, test_idx)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(trainFaces,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testFaces,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    model = Net(args).to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=args.gamma)

    test_loss = []
    test_acc = []
    train_loss = []

    for epoch in range(1, args.epochs + 1):
        train_loss.append(train(args, model, device, train_loader, optimizer, epoch))
        results = test(args, model, device, test_loader)
        test_loss.append(results[0])
        test_acc.append(results[1])
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "yale_cnn.pth")
        
    return test_acc[-1]

def run_train_net_kfold(args):
    yaleData = ImageFolder('~/PycharmProjects/summer/data/CroppedYale/',
                       transform=transforms.Compose([
                           transforms.Grayscale(),
                           transforms.Resize((192,168), interpolation=0),
                           transforms.ColorJitter(brightness=.5, contrast=.3), # Random recolorization every load.
                           transforms.ToTensor()
                       ]))
    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    results = []
    for i, (train_idx, test_idx) in enumerate(kfold.split(yaleData)):
        test_accuracy = run_train_net_once(yaleData, train_idx, test_idx, args)
        results.append(test_accuracy)
    
    mean = np.mean(results)
    variance = np.std(results) ** 2
    print(f"Result: {np.mean(results)} +- {np.std(results)} ")
    return mean, variance    
    

if __name__ == "__main__":
    args = Namespace(
        no_cuda=False, 
        seed=1, 
        batch_size=2,
        test_batch_size=1000,
        epochs=23,
        lr=1.0,
        gamma=0.7,
        log_interval=250, # was 250
        save_model=False,
        num_folds=2
    )
    run_train_net_kfold(args)