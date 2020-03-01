# create data pipeline to split the training data
import time
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
from torch.utils.data import DataLoader, sampler
from torchsummary import summary
from sklearn.metrics import accuracy_score
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torchvision import models, transforms, datasets
import os
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from time import time
import pandas as pd
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread
import json  # create the json
import shutil  # copy images to train, test and valid dirs
import os  # files and dirs manipulation
import math  # split calculate
# path configuration (can be set up in any directory based on user environment,here we have set it up
# in the google lab
# Please note we have used google lab environment to leverage cuda performance to train below models
from google.colab import drive
drive.mount('/content/drive')
parent_dir = '/content/drive/My Drive/thecarconnectionpicturedataset2'
# we have manually created two folders in the above directory to store all audi images in the Audi folder
# and to store all bmw images in the BMW folder
os.chdir(parent_dir)
category_list = list(filter(lambda x: os.path.isdir(x), os.listdir()))
for category in category_list:
    print(category)
# create training,validation,testing directories
data_set_dirs = ['train', 'valid', 'test']
for dsdirs in data_set_dirs:
  path = parent_dir + '/' + dsdirs
  os.mkdir(path, 755)
# define proportion of data
train_prop = 0.6
valid_prop = test_prop = (1-train_prop)/2
# function to split data of each category into trainning, validation and testing set


def create_dataset():
  for ii, cat in enumerate(category_list):
    src_path = parent_dir + '/' + cat
    dest_dir1 = parent_dir+'/train/'+str(ii)
    dest_dir2 = parent_dir+'/valid/'+str(ii)
    dest_dir3 = parent_dir+'/test/'+str(ii)

    dest_dirs_list = [dest_dir1, dest_dir2, dest_dir3]
    for dirs in dest_dirs_list:
      os.mkdir(dirs, 755)

    # get files' names list from respective directories
    os.chdir(src_path)
    files = [f for f in os.listdir() if os.path.isfile(f)]

    # get training, testing and validation files count
    train_count = math.ceil(train_prop*len(files))
    valid_count = int((len(files)-train_count)/2)
    test_count = valid_count

    # get files to segragate for train,test and validation data set
    train_data_list = files[0: train_count]
    valid_data_list = files[train_count+1:train_count+1+valid_count]
    test_data_list = files[train_count+valid_count:]

    for train_data in train_data_list:
      train_path = src_path + '/' + train_data
      shutil.copy(train_path, dest_dir1)

    for valid_data in valid_data_list:
      valid_path = src_path + '/' + valid_data
      shutil.copy(valid_path, dest_dir2)

    for test_data in test_data_list:
      test_path = src_path + '/' + test_data
      shutil.copy(test_path, dest_dir3)


# save category data as dictionary in a json file
cat_data = {}
for ix, cat in enumerate(category_list):
  cat_data[ix] = cat
with open('/content/drive/My Drive/thecarconnectionpicturedataset2/cat_data.json', 'w') as outfile:
    json.dump(cat_data, outfile)

# Here we use cuda model to train fully connected layers for two pre-trained models,
# which is vgg16 and Alexnet
InteractiveShell.ast_node_interactivity = 'all'

# Location of data
dataset = 'thecarconnectionpicturedataset2'
datadir = '/content/drive/My Drive/thecarconnectionpicturedataset2/'
traindir = datadir + 'train'
validdir = datadir + 'valid'
testdir = datadir + 'test'

# preprocess the images
image_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}

dataloaders = {
    'train': DataLoader(data['train'], shuffle=True, batch_size=128),
    'val': DataLoader(data['val'], shuffle=True, batch_size=128),
    'test': DataLoader(data['test'], shuffle=True, batch_size=128)
}
train_data_size = len(data['train'])
valid_data_size = len(data['val'])
test_data_size = len(data['test'])


train_data_loader = dataloaders['train']
valid_data_loader = dataloaders['val']
test_data_loader = dataloaders['test']

# loading the pretrained model vgg16
model_1 = models.vgg16(pretrained=True)
# Freeze model weights
for parameter in model_1.parameters():
    parameter.requires_grad = False
# checking if GPU is available if yes, then running on gpu
if torch.cuda.is_available():
    model_1 = model_1.cuda()


# specify loss function (categorical cross-entropy)
loss_criterion = nn.NLLLoss()

# specify optimizer (stochastic gradient descent) and learning rate
# gradient descent on the layer that we customized in transfer learning
optimizer = optim.Adam(model_1.classifier.parameters())

# define the function to train and validate the model


def train_and_validate(model, loss_criterion, optimizer, epochs=20):
      start = time.time
  history=[]
  best_acc=0.0
  

  for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))

    model.train()

    train_loss=0.0
    train_acc=0.0
    valid_loss=0.0
    valid_acc=0.0
    for i,(inputs,labels) in enumerate(train_data_loader):
      inputs=inputs.cuda()
      labels=labels.cuda()
      optimizer.zero_grad()
      outputs=model(inputs)
      loss=loss_criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * inputs.size(0)
      ret, predictions = torch.max(outputs.data, 1)
      correct_counts = predictions.eq(labels.data.view_as(predictions))
      acc = torch.mean(correct_counts.type(torch.FloatTensor))
      train_acc += acc.item() * inputs.size(0)
    with torch.no_grad():
      model.eval()
      for j, (inputs, labels) in enumerate(valid_data_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)
        valid_loss += loss.item() * inputs.size(0)
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        valid_acc += acc.item() * inputs.size(0)
    avg_train_loss = train_loss/train_data_size
    avg_train_acc = train_acc/train_data_size
    avg_valid_loss = valid_loss/valid_data_size 
    avg_valid_acc = valid_acc/valid_data_size
    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
    epoch_end = time.time()
    print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
    torch.save(model, dataset+'_modelalex_'+str(epoch)+'.pt')
   
    

  return model, history

# import time
trained_model, history = train_and_validate(model_1, loss_criterion, optimizer, epochs=30)
# create a saving point for the model
torch.save(history, dataset+'_history.pt')

# build the diagrams to demonstrate the relationship between training and validation sets

# training loss and validation loss over epochs 
history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()
# train accuracy and validation accuracy 
plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()

def computeTestSetAccuracy(model, loss_criterion):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))

# execute the function by using the best saved model
bestvgg = torch.load('thecarconnectionpicturedataset2_model_11.pt')
computeTestSetAccuracy(bestvgg, loss_criterion)

# loading pre-trained model Alexnet
model_2 = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=True)

# Freeze model weights
for param in model_2.parameters():
    param.requires_grad = False

# Add on classifier
model_2.classifier[6] = nn.Sequential(
    nn.Linear(4096, 256), nn.ReLU(),
    nn.Linear(256, 2),nn.LogSoftmax(dim=1))

for param in model_2.classifier[6].parameters():
    param.requires_grad = True

# checking if GPU is available if yes, then running on gpu
if torch.cuda.is_available():
    model_2 = model_2.cuda()

import torch.optim as optim

# specify loss function (categorical cross-entropy)
loss_criterion = nn.NLLLoss()

# specify optimizer (stochastic gradient descent) and learning rate
# gradient descent on the layer that we customized in transfer learning
optimizer = optim.Adam(model_2.classifier.parameters())

# use the function to train fully connected layer of alexnet 
trained_model_alex, history_alex = train_and_validate(model_2, loss_criterion, optimizer, epochs=30)
torch.save(history_alex, dataset+'_alexhistory.pt')

# build the diagrams to demonstrate the relationship between training and validation sets

# train loss and validation loss
history_alex = np.array(history_alex)
plt.plot(history_alex[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.title('Loss Comparison For AlexNet')
plt.savefig(dataset+'_loss_curve_AlexNet.png')
plt.show()

# train accuracy and validation accuracy
plt.plot(history_alex[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.title('Accuracy Comparison For AlexNet')
plt.savefig(dataset+'_accuracy_curve_AlexNet.png')
plt.show()

# saved the best model and compute the accuracy
bestalex = torch.load('thecarconnectionpicturedataset2_modelalex_15.pt')
computeTestSetAccuracy(bestalex, loss_criterion)

# construct an general CNN from scratch without using the pre-trained model 
class generic(nn.Module):
      def __init__(self):
    super(generic, self).__init__()

    self.cnn_layers = Sequential(
        nn.Conv2d(3,64,kernel_size=5,stride=1,padding=2),
        nn.ReLU(),
        MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(64,128,kernel_size=5,stride=1,padding=2),
        nn.ReLU(),
        MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(128,256,kernel_size=5,stride=1,padding=2),
        nn.ReLU(),
        MaxPool2d(kernel_size=2,stride=2))
    
    self.drop_out = nn.Dropout()
    self.linear_layers=Sequential(
        Linear(28*28*256,500),
        Linear(500,250),
        Linear(250,2))

  def forward(self,x):
    x=self.cnn_layers(x)
    x=x.view(x.size(0),-1)
    x=self.drop_out(x)
    x=self.linear_layers(x)
    return x

# defining the generic model
model_generic = generic()
# defining the optimizer
optimizer = Adam(model_generic.parameters())
# defining the loss function
loss_criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
  model_generic = model_generic.cuda()
  loss_criterion = loss_criterion.cuda()

print(model_generic)

# train the model and save it 
trained_generic_model, history_generic = train_and_validate(model_generic, loss_criterion, optimizer, epochs=20)
torch.save(history_generic, dataset+'_generichistory.pt')

# build the diagrams to demonstrate the relationship between training and validation sets

# train loss and validation loss
history_generic = np.array(history_generic)
plt.plot(history_generic[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.title('Loss Comparison For Generic Model')
plt.savefig(dataset+'_loss_curve_genericmodel.png')
plt.show()

# train accuracy and validation accuracy
plt.plot(history_generic[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.title('Accuracy Comparison For Generic Model')
plt.savefig(dataset+'_accuracy_curve_genericmodel.png')
plt.show()

# saved the best model and compute the accuracy
bestgeneric = torch.load('thecarconnectionpicturedataset2_modelgeneric_7.pt')
computeTestSetAccuracy(bestgeneric, loss_criterion)




