from sklearn.metrics import f1_score
import copy
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import random
import numpy as np
import argparse

import scipy.io as scio

import pickle
import torch.nn.functional as F
import os

data_list=[]
label_list=[]

labels=set()
root=r'/home/jeff/demo/pytorch-flowers-classification/flowers'
labels=sorted(os.listdir(root))
label_dict=dict()
for i in range(len(labels)):
    label_dict[labels[i]]=i

import cv2

def getPathList(root):
    items = os.listdir(root)
    pathList=[]
    for item in items:
        full_path=os.path.join(root, item)
        if os.path.isdir(full_path):
            pathList.extend(getPathList(full_path))
        else:
            pathList.append(full_path)

    return pathList

pathList = getPathList(root)

for path in pathList:
    flower_name=path.split(os.path.sep)[-2]
    img=cv2.imread(path)

    data_list.append(img)
    label_list.append(label_dict[flower_name])

features, labels= data_list, label_list
dim_1=39
dim_2=100

SEED=336
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
np.random.seed(SEED)

feature_vector_filename='feature_vector.pkl'
label_dict_filename='label_dict.pkl'

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, feature_list, label_list):
        self.feature_list = feature_list
        self.label_list = label_list

    def __getitem__(self, index):

        self.feature_list[index] = cv2.resize(self.feature_list[index], (100, 100))/255.

        feature = torch.from_numpy(self.feature_list[index]).float().permute(2,0,1)

        label = torch.from_numpy(np.asarray(self.label_list[index])).long()

        return feature, label

    def __len__(self):
        return len(self.label_list)

test_size=0.2

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=7, shuffle=True)

n_train=int(len(X_train)*0.7)

X_dev = X_train[n_train:]
y_dev = y_train[n_train:]

X_train=X_train[:n_train]
y_train = y_train[:n_train]

print(len(X_train), len(X_dev), len(X_test))

def get_label_cnt_map(y):

    label_cnt_map=dict()
    for i in range(len(y)):
        if y[i] not in label_cnt_map:
            label_cnt_map[y[i]]=1
        else:
            label_cnt_map[y[i]]+=1
    return label_cnt_map

print('===train label count===')
label_cnt_map=get_label_cnt_map(y_train)
print(label_cnt_map)
print("=====test label count=====")
label_cnt_map=get_label_cnt_map(y_test)
print(label_cnt_map)
# exit()

train_data = MyDataset(X_train, y_train)
dev_data = MyDataset(X_dev, y_dev)
test_data = MyDataset(X_test, y_test)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 200

batch_size = 32

learning_rate = 1e-3

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(1,1)),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc1 =nn.Sequential()
        self.fc1.add_module('fc_layer1',nn.Linear(28224, 2400))
        self.fc1.add_module('fc_layer2', nn.Linear(2400, num_classes))
        # self.fc1.add_module('fc_layer_2_activation', nn.Softmax(dim=1))

    def convNet(self, x):

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        x=x.reshape(x.size(0), -1)

        x=self.fc1(x)

        return x

    def forward(self, x):
        conv_feat=self.convNet(x)

        return conv_feat

# Data loader
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
dev_data_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = MyModel(5).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

saved_model_name='model.ckpt'

curr_patience = patience = 6
best_valid_loss=float('inf')

num_trials=1

for epoch in range(num_epochs):
    model.train()

    total_step = len(train_data_loader)

    for i, (features, labels) in enumerate(train_data_loader):

        model.zero_grad()

        features = features.to(device)

        labels = labels.to(device)

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], 1.0)
        optimizer.step()

        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                     total_step, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        epoch_loss=0.0
        epoch_f1_score=0.0

        for features, labels in dev_data_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)
            epoch_loss+=loss

            # labels=torch.argmax(labels.data, 1)
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        epoch_loss/=len(dev_data_loader)
        valid_loss=epoch_loss
        valid_acc=100 * (correct / total)
    valid_loss=epoch_loss
    valid_acc= 100 * (correct / total)
    print('valid_loss: {:.3f} | valid_acc: {:.3f}'.format(valid_loss, valid_acc))
    print('current patience: {}'.format(curr_patience))
    if valid_loss<=best_valid_loss:
        print('Found new best model on dev set.')
        torch.save(model.state_dict(), 'model.ckpt')
        torch.save(optimizer.state_dict(), 'optim_best.std')
        best_valid_loss=valid_loss
        curr_patience = patience
    else:
        curr_patience -= 1
        if curr_patience <=-1:
            print('Running out of patience, loading previous best model.')
            num_trials -= 1
            curr_patience = patience
            model.load_state_dict(torch.load('model.ckpt'))
            optimizer.load_state_dict(torch.load('optim_best.std'))
            # lr_scheduler.step()
    if num_trials<=0:
        print('Running out of patience, training stops!')
        model.load_state_dict(torch.load('model.ckpt'))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            epoch_loss=0.0
            epoch_f1_score=0.0

            for features, labels in test_data_loader:
                features = features.to(device)
                labels = labels.to(device)

                # epoch_loss, correct, total, epoch_f1_score=eval_help(features, labels, epoch_loss, epoch_f1_score, total, correct)
                outputs = model(features)

                loss = criterion(outputs, labels)
                epoch_loss+=loss

                # labels=torch.argmax(labels.data, 1)
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


            epoch_loss/=len(dev_data_loader)

        test_loss=epoch_loss
        test_acc=100 * (correct / total)
        print('test_loss: {:.3f} | test_acc: {:.3f}'.format(test_loss, test_acc))
        exit()

model.load_state_dict(torch.load('model.ckpt'))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    epoch_loss=0.0
    epoch_f1_score=0.0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        epoch_loss, correct, total=eval_help(features, labels, epoch_loss, epoch_f1_score, total, correct)

    epoch_loss/=len(dataloader)

test_loss=epoch_loss
test_acc=100 * (correct / total)
print('test_loss: {:.3f} | test_acc: {:.3f}'.format(test_loss, test_acc))







#
