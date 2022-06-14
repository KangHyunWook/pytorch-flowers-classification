"""
"""
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torch.nn as nn
import torch

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
    
model = ConvNet(5)
model = nn.DataParallel(model).cuda()   

model.load_state_dict(torch.load('saved_model.pth'))

normalize=transforms.Normalize(mean=[0.57535914, 0.44928582, 0.40079932],
                                          std=[0.20735591, 0.18981615, 0.18132027])

test_dir=r'C:\data\out-flowers\test'

test_dataset = datasets.ImageFolder(test_dir, 
                                        transforms.Compose([transforms.Resize((50,50)),
                                        transforms.ToTensor(),
                                        normalize]))
test_loader = DataLoader(test_dataset, batch_size = 30, shuffle=False)

acc, err = test(model, test_loader)

print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
