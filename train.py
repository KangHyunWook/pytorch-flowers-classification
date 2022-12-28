from torch import optim

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torch.nn as nn
import torch
import os
import torch.backends.cudnn as cudnn
import numpy as np

from torch.optim import lr_scheduler

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.prelu = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)

        self.fc1 = nn.Linear(128*6*6, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x=self.prelu(self.conv1_1(x))
        x=self.prelu(self.conv1_2(x))
        x=F.max_pool2d(x, 2)

        x=self.prelu(self.conv2_1(x))
        x=self.prelu(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x= self.prelu(self.conv3_1(x))
        x=self.prelu(self.conv3_2(x))
        x=F.max_pool2d(x, 2)
        # print('==x shape==')
        # print(x.shape)
        # exit()
        x=x.view(-1, 128*6*6)
        x=self.prelu(self.fc1(x))
        y=self.fc2(x)

        return y

def test(model, val_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total+= labels.size(0)
            correct+=(predictions == labels.data).sum()

    acc = correct *100. / total
    err = 100. - acc

    return acc, err

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, feature_list, label_list):
        self.feature_list = feature_list
        self.label_list = label_list
        self.desire_shape = [50,50]

    def __getitem__(self, index):

        self.feature_list[index] = cv2.resize(self.feature_list[index],(50,50))/255.

        feature = torch.from_numpy(self.feature_list[index]).float()
        label = torch.from_numpy(np.asarray(self.label_list[index])).long()

        return feature, label
    def __len__(self):
        return len(self.label_list)

root=r'/home/jeff/demo/pytorch-flowers-classification/flowers'

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
print(len(pathList))

data_list=[]
label_list=[]

labels=set()

labels=sorted(os.listdir(root))
label_dict=dict()
for i in range(len(labels)):
    label_dict[labels[i]]=i

import cv2

for path in pathList:
    flower_name=path.split(os.path.sep)[-2]
    img=cv2.imread(path)

    data_list.append(img)
    label_list.append(label_dict[flower_name])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_list, label_list, test_size=0.2, random_state=7)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=7)

train_data = MyDataset(X_train, y_train)
dev_data = MyDataset(X_val, y_val)

print(len(X_train), len(X_val), len(X_test))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
dev_data_loader = DataLoader(dev_data, batch_size = 32, shuffle=False)

import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_sizes = [128*2, 128]

        self.conv1=nn.Sequential()
        self.conv1.add_module('conv_layer1', nn.Conv2d(3, 16, kernel_size=3))
        self.conv1.add_module('conv_layer1_activation', nn.ReLU())

        self.conv2= nn.Sequential()
        self.conv2.add_module('conv_layer2', nn.Conv2d(16, 32, kernel_size=3))
        self.conv2.add_module('conv_layer2_activation', nn.ReLU())

        self.fc3=nn.Sequential()
        self.fc3.add_module('fc_layer1',nn.Linear(32*46*46, 5))
        # self.fc3.add_module('fc_layer1_activatoin', nn.Softmax(dim=1))

    def forward(self, x):

        x=x.permute(0, 3, 1, 2)

        x=self.conv1(x)
        x=self.conv2(x)

        x=x.reshape(-1, 32*46*46)

        x=self.fc3(x)
        # print(x.shape)
        # exit()
        return x

model = MyModel().to(device)

criterion= nn.CrossEntropyLoss(reduction='mean')

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

for epoch in range(30):
    model.train()
    for i, (features, labels) in enumerate(train_data_loader):
        model.zero_grad()
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        loss= criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], 1.0)
        optimizer.step()
        if i%10==0:
            print('Epoch {}/30 | Loss: {:.4f}'.format(epoch+1, loss.item()))
            # valid_loss, valid_acc = evaluate(dev_data_loader)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        epoch_loss = 0.0

        epoch_acc=0
        for features, labels in dev_data_loader:
            features= features.to(device)
            labels = labels.to(device)
            outputs=model(features)
            loss=criterion(outputs, labels)
            print(outputs)
            predicted=torch.argmax(outputs.data,1)

            correct=((predicted==labels).sum()).item()

            epoch_loss+=loss
            epoch_acc+=correct/features.size(0)

        epoch_loss/=len(dev_data_loader)
        epoch_acc/=len(dev_data_loader)
    print('valid_loss: {:.4f} | valid_acc: {:.3f}'.format(epoch_loss, epoch_acc))





















#
