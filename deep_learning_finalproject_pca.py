# import required libraries
import math  # split calculate
import shutil  # copy images to train, test and valid dirs
import cv2 as cv
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder
import os
from torchvision import models, transforms, datasets
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score
from torchsummary import summary
from torch.utils.data import DataLoader, sampler
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# path configuration (can be set up in any directory based on user environment,here we have set it up
# in the google lab
# Please note we have used google lab environment to leverage cuda performance to train below models
drive.mount('/content/drive')
parent_dir = '/content/drive/My Drive/thecarconnectionpicturedataset'
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


create_dataset()

# after we have created the folders with train,valid, and test datasets
cd '/content/drive/My Drive/thecarconnectionpicturedataset2/train/1'
files = glob('*.jpg')
files = np.random.permutation(files)
train_img = []
for img_name in tqdm(files):
    img = imread(img_name)
    train_img.append(img)


def pca(x):
      result = []
  for i in tqdm(x):
    renorm_image = np.reshape(i,(i.shape[0]*i.shape[1],3))
    mean = np.mean(renorm_image, axis=0)                           #computing the mean
    std = np.std(renorm_image, axis=0)                             #computing the standard deviation
    renorm_image = renorm_image.astype('float32')                  #we change the datatpe so as to avoid any warnings or errors
    renorm_image -= np.mean(renorm_image, axis=0)                  
    renorm_image /= np.std(renorm_image, axis=0)                   # next we normalize the data using the 2 columns
    cov = np.cov(renorm_image, rowvar=False) 
    lambdas, p = np.linalg.eig(cov)
    alphas = np.random.normal(0, 0.1, 3)
    delta = np.dot(p, alphas*lambdas)
    pca_augmentation_version_renorm_image = renorm_image + delta    #forming augmented normalised image
    pca_color_image = pca_augmentation_version_renorm_image * std + mean             #de-normalising the image
    pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype('uint8')  # necessary conditions which need to be checked
    pca_color_image=np.ravel(pca_color_image).reshape((i.shape[0],i.shape[1],3))                                          
    result.append(pca_color_image)
  return result
  pca_iam=pca(train_img)

os.mkdir('/content/drive/My Drive/thecarconnectionpicturedataset2/train/11')

# compare the image befroe and after PCA 
plt.imshow(train_img[5])
plt.imshow(pca_iam[5])

from PIL import Image

img = Image.fromarray(array)
img.save('testrgb.png')
cd '/content/drive/My Drive/thecarconnectionpicturedataset2/train/11'
import PIL
for i in range(len(pca_iam)):
  img = PIL.Image.fromarray(pca_iam[i])
  img.save(str(i)+'.png')

cd '/content/drive/My Drive/thecarconnectionpicturedataset

# Location of data
dataset='thecarconnectionpicturedataset'
datadir = '/content/drive/My Drive/thecarconnectionpicturedataset/'
traindir = datadir + 'trainpca'
validdir = datadir + 'valid'
testdir = datadir + 'test'

image_transforms={
    'train':
    transforms.Compose([
        transforms.Resize(size=256),          
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  #removed color jogger because we've already preprocessed pictures by PCA
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
    'test' :
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

data={
    'train':
    datasets.ImageFolder(root=traindir,transform=image_transforms['train']),
    'val':
    datasets.ImageFolder(root=validdir,transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir,transform=image_transforms['test'])
}

dataloaders= {
    'train': DataLoader(data['train'],shuffle=True,batch_size=128),
    'val' : DataLoader(data['val'],shuffle=True,batch_size=128),
    'test' : DataLoader(data['test'],shuffle=True,batch_size=128)
}
train_data_size = len(data['train'])
valid_data_size = len(data['val'])
test_data_size = len(data['test'])


train_data_loader = dataloaders['train']
valid_data_loader = dataloaders['val']
test_data_loader = dataloaders['test']

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
features.shape, labels.shape

# loading the pretrained model
model_1 = models.vgg16(pretrained=True)
# Freeze model weights
for parameter in model_1.parameters():
    parameter.requires_grad = False
# Add on classifier
model_1.classifier[6] = nn.Sequential(
    nn.Linear(4096, 256), nn.ReLU(),
    nn.Linear(256, 2),nn.LogSoftmax(dim=1))

for parameter in model_1.classifier[6].parameters():
    parameter.requires_grad = True
# checking if GPU is available if yes, then running on gpu
if torch.cuda.is_available():
    model_1 = model_1.cuda()

import torch.optim as optim

# specify loss function (categorical cross-entropy)
loss_criterion = nn.NLLLoss()

# specify optimizer (stochastic gradient descent) and learning rate
# gradient descent on the layer that we customized in transfer learning
optimizer = optim.Adam(model_2.classifier.parameters())

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
    torch.save(model, dataset+'_alexnetwithpca_'+str(epoch)+'.pt')
    
    

  return model, history

import time
trained_model, history = train_and_validate(model_1, loss_criterion, optimizer, epochs=20)
torch.save(history, dataset+'_vggpcahistory.pt')

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

bestvggpca = torch.load('thecarconnectionpicturedataset2_modelwithpca_17.pt')
computeTestSetAccuracy(bestvggpca, loss_criterion)

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
import time
trained_model_pca_alex, history_pca_alex = train_and_validate(model_2, loss_criterion,
optimizer, epochs=20)

torch.save(history_pca_alex, dataset+'_ALEXpcahistory.pt')

# build the diagrams to demonstrate the relationship between training and validation sets

# training loss and validation loss over epochs 

history_pca_alex = np.array(history_pca_alex)
plt.plot(history_pca_alex[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.title('Loss Comparison for AlexNet with PCA')
plt.savefig(dataset+'_loss_curve.png')
plt.show()

# training accuracy and validation accuracy
plt.plot(history_pca_alex[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for AlexNet with PCA')
plt.ylim(0,1)
plt.savefig(dataset+'_alexpca_accuracy_curve.png')
plt.show()

bestalexpca = torch.load('thecarconnectionpicturedataset2_alexnetwithpca_5.pt')
computeTestSetAccuracy(bestalexpca, loss_criterion)
