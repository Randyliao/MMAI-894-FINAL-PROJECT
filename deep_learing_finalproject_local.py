# -*- coding: utf-8 -*-
"""
Move files over into train, validate and test folders
"""

import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
import shutil
from datetime import date

print(os.getcwd())

# set up the directory based on local environment,
# create a folder that unzips all images of Audi and BMW without interior
filepath = 'car_images_no_int_BMW_Audi'

bmw_files = glob.glob(filepath+"/BMW_*.jpg")
num_bmw = len(bmw_files)

print("Number of BMW files: " + str(num_bmw))

audi_files = glob.glob(filepath+"/Audi_*.jpg")
num_audi = len(audi_files)

print("Number of Audi files: " + str(num_audi))

num_to_process = 0
if num_audi < num_bmw:
    num_to_process = num_audi
else:
    num_to_process = num_bmw

print("Number of files to process: " + str(num_to_process))

train_test_ratio = 0.6

num_files_trainset = int(num_to_process * train_test_ratio)
num_files_valset = int((num_to_process-num_files_trainset)/2)

print("Number of files that go into training set: " + str(num_files_trainset))


# randomly shuffle lists
random.shuffle(audi_files)
random.shuffle(bmw_files)

dest_root = "images_" + str(date.today())
os.mkdir(dest_root)
os.mkdir(dest_root + "/train")
os.mkdir(dest_root + "/validation")
os.mkdir(dest_root + "/test")

path_to_train_bmw = dest_root + "/train/bmw"
path_to_train_audi = dest_root + "/train/audi"
path_to_val_bmw = dest_root + "/validation/bmw"
path_to_val_audi = dest_root + "/validation/audi"
path_to_test_bmw = dest_root + "/test/bmw"
path_to_test_audi = dest_root + "/test/audi"

os.mkdir(path_to_train_bmw)
os.mkdir(path_to_train_audi)
os.mkdir(path_to_val_bmw)
os.mkdir(path_to_val_audi)
os.mkdir(path_to_test_bmw)
os.mkdir(path_to_test_audi)

for i in range(num_to_process):
    bmw_filename = bmw_files[i].split('\\')[1]
    audi_filename = audi_files[i].split('\\')[1]
    if i < num_files_trainset:
        # move bmw image to bmw training folder
        shutil.move(filepath + "/" + bmw_filename,
                    path_to_train_bmw + "/" + bmw_filename)
        # move audi image to audi training folder
        shutil.move(filepath + "/" + audi_filename,
                    path_to_train_audi + "/" + audi_filename)
    elif i >= num_files_trainset and i < (num_files_trainset + num_files_valset):
        # move bmw image to bmw validation folder
        shutil.move(filepath + "/" + bmw_filename,
                    path_to_val_bmw + "/" + bmw_filename)
        # move audi image to audi validation folder
        shutil.move(filepath + "/" + audi_filename,
                    path_to_val_audi + "/" + audi_filename)
    else:
        # move bmw image to bmw validation folder
        shutil.move(filepath + "/" + bmw_filename,
                    path_to_test_bmw + "/" + bmw_filename)
        # move audi image to audi validation folder
        shutil.move(filepath + "/" + audi_filename,
                    path_to_test_audi + "/" + audi_filename)

# Transfer Learning for Recognizing BMW vs Audi images using ResNet

# EarlyStopping Class adapted from https://github.com/Bjarten/early-stopping-pytorch
# This class will be used in our main training and testing loop to break out of training early if necessary


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# PyTorch Data Generators to easily get and transform images into tensors
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ]),
    'test':
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])
}

date_str = '_2020-02-28'

image_datasets = {
    'train':
    datasets.ImageFolder('images' + date_str + '/train',
                         data_transforms['train']),
    'validation':
    datasets.ImageFolder('images' + date_str + '/validation',
                         data_transforms['validation']),
    'test':
    datasets.ImageFolder('images' + date_str + '/test',
                         data_transforms['test'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=128,
                                shuffle=True, num_workers=4),
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=128,
                                shuffle=False, num_workers=4),
    'test':
    torch.utils.data.DataLoader(image_datasets['test'],
                                batch_size=128,
                                shuffle=False, num_workers=4)
}

# Augment the ResNet Neural Network with our own classifier
# using CPU in this case
device = torch.device("cpu")

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2)).to(device)

for parameter in model.fc.parameters():
    parameter.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Train Model
# Changed slightly for early stopping
# Loads best checkpoint


def train_model(model, criterion, optimizer, num_epochs=3, patience=3):
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    valid_loss = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{} , '.format(epoch+1, num_epochs), end=" ")

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(
                phase, epoch_loss.item(), epoch_acc.item()))
            print(' ')

            if phase == 'train':
                train_loss_arr.append(epoch_loss.item())
                train_acc_arr.append(epoch_acc.item())
            else:
                valid_loss = epoch_loss.item()
                val_loss_arr.append(epoch_loss.item())
                val_acc_arr.append(epoch_acc.item())

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    history = {'train_loss': train_loss_arr, 'train_acc': train_acc_arr,
               'val_loss': val_loss_arr, 'val_acc': val_acc_arr}
    return model, history


model_trained, history = train_model(
    model, criterion, optimizer, num_epochs=50, patience=3)

# Save Model for Later Use
torch.save(model_trained.state_dict(),
           'savedmodels/pytorch/ResNetTL-bmwVSaudi-v2.h5')
#Training and Validation Loss and Accuracy by Epoch
# Training loss verus validation loss
plt.figure(figsize=(8, 6))
plt.plot(history['train_loss'], label="Training Loss")
plt.plot(history['val_loss'], label="Validation Loss")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Transfer Learning with ResNet50: Loss by Epoch')

#Training accuracy 
plt.figure(figsize=(8, 6))
plt.plot(history['train_acc'], label="Training Accuracy")
plt.plot(history['val_acc'], label="Validation Accuracy")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Transfer Learning with ResNet50: Accuracy by Epoch')

#Test the Trained Model
def test_set_accuracy(model, loss_criterion, test_data_loader, test_data_size, device_type="cpu"):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    device = torch.device("cpu")
    if device_type == "cuda":
        device = torch.device("cuda:0")

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

model.load_state_dict(torch.load('checkpoint.pt'))
test_set_accuracy(model, criterion, dataloaders['test'], len(image_datasets['test']))
