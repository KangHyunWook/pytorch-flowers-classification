from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn import functional as F

import torch.nn as nn
import torch
import os
import torch.backends.cudnn as cudnn
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
if __name__=='__main__':
      
    train_dir=r'C:\data\out-flowers\train'
    val_dir=r'C:\data\out-flowers\val'

    normalize=transforms.Normalize(mean=[0.57535914, 0.44928582, 0.40079932],
                                          std=[0.20735591, 0.18981615, 0.18132027])


    train_dataset = datasets.ImageFolder(train_dir, 
                                            transforms.Compose([transforms.RandomResizedCrop((50,50)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor(),
                                                                normalize])) 
    
    val_dataset = datasets.ImageFolder(val_dir,
                                            transforms.Compose([transforms.Resize((50,50)),
                                                                transforms.ToTensor(),
                                                                normalize]))
    
    trainloader=DataLoader(dataset=train_dataset, shuffle=True, batch_size=30)
    val_loader=DataLoader(dataset=val_dataset, shuffle=False, batch_size=30)
    
    
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
     
    cudnn.benchmark=True
    torch.cuda.manual_seed_all(1)
    
    model=ConvNet(5)
    model = nn.DataParallel(model).cuda()   

    criterion_xent = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 5e-04, momentum=0.9)
    #todo    
    scheduler = lr_scheduler.StepLR(optimizer_model, step_size=20, gamma=0.5)
    
    for epoch in range(100):
        print("==> Epoch {}/100".format(epoch+1))
        
        model.train()
        xent_losses = AverageMeter()
        
        for batch_idx, (data, labels) in enumerate(trainloader):
           
            data, labels = data.cuda(), labels.cuda()
            outputs=model(data)
            
            loss_xent = criterion_xent(outputs, labels)
            optimizer_model.zero_grad()
            
            loss_xent.backward()
            optimizer_model.step()
            
            xent_losses.update(loss_xent.item(), labels.size(0))
            
            if (batch_idx+1) % 50 == 0:
                print("Batch {}/{}\t XentLoss {:.6f} ({:.6f}) " \
                        .format(batch_idx+1, len(trainloader), xent_losses.val, xent_losses.avg))
                        
        print('===current lr====')
        for key in optimizer_model.param_groups:
            print(key['lr'])
            
        scheduler.step()
        
        if (epoch+1) % 10 == 0 or (epoch+1) ==100:
            print("==> Test")
            acc, err = test(model, val_loader)
            print('Accuracy (%): {}\t Error rate (%): {}'.format(acc, err))
            
    torch.save(model.state_dict(), 'saved_model.pth')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    