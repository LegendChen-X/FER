import matplotlib.pyplot as plt
import numpy as np
import torch, os
import pandas as pd
import torch.nn as nn
import math
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from util import *

# We have reference for the ResNet18 class from the link: https://github.com/Nebula4869/PyTorch_facial_expression_recognition/blob/master/model.py
# We also reuse some of code from our previous project in CSC420.

classes = {
    0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Anger', 6: 'Neutral'
}


def conv_block(in_chnl, out_chnl, pool=False, padding=1):
    layers = []
    layers.append(nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding))
    layers.append(nn.BatchNorm2d(out_chnl))
    layers.append(nn.ReLU(inplace=True))
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
    

class Base(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validating(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

class ResNet09(Base):
    def __init__(self, in_chnls, num_cls):
        super().__init__()
        self.conv1 = conv_block(in_chnls, 64, pool=True)
        self.conv2 = conv_block(64, 128, pool=True)
        self.resnet1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.resnet2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(3), nn.Flatten(), nn.Linear(512, num_cls))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.resnet1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.resnet2(out) + out
        return self.classifier(out)
        
        
class VGG(Base):
    def __init__(self, in_chnls, num_cls):
        super(VGG, self).__init__()
        vgg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        layers = []
        in_channels = in_chnls
        for x in vgg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_cls)
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

class Dataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.inputs = images
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        data = self.inputs[i]
        data = np.asarray(data).astype(np.uint8).reshape(48,48,1)
        data = self.transforms(data)
        label = self.labels[i]
        return (data, label)
        

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
        
def evaluate(model, val_loader):
    model.eval()
    #print(model)
    outputs = [model.validating(batch) for batch in val_loader]
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
        
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down_sample = down_sample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out
        
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validating(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

class ResNet18(Base):

    def __init__(self, block, layers, num_classes=1000):
        self.in_planes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, down_sample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
def fit(epochs, max_lr, model, train_loader, val_loader, weight_decay, grad_clip, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            if grad_clip: nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))
        history.append(result)
    return history
    
    
def main(type, batch, epoch):
    data = np.load("./data/data.npz")
    print("Load the data file successfully, start training:")

    train_images = data["train_images"]
    train_labels = data["train_labels"]
    test_images = data["test_images"]
    test_labels = data["test_labels"]

    train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(48, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5), inplace=True)
    ])

    valid_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    train_data = Dataset(train_images, train_labels, train_transform)
    valid_data = Dataset(test_images, test_labels, valid_transform)
    print("Training and validation dataset initialization finished")
    torch.manual_seed(413)
    batch_num = batch
    trainDataLoader = DataLoader(train_data, batch_num, shuffle=True, num_workers=4, pin_memory=True)
    validDataLoader = DataLoader(valid_data, batch_num*2, num_workers=4, pin_memory=True)
    print("TrainDataLoader and validDataLoader initialization finished")
    device = get_default_device()
    trainDataLoader = DeviceDataLoader(trainDataLoader, device)
    validDataLoader = DeviceDataLoader(validDataLoader, device)
    if type=="VGG":
        model = to_device(VGG(1, 7), device)
    elif type=="ResNet09":
        model = to_device(ResNet09(1, 7), device)
    elif type=="ResNet18":
        model = to_device(ResNet18(BasicBlock, [2,2,2,2], 7), device)
    else:
        model1 = to_device(VGG(1, 7), device)
        model2 = to_device(ResNet09(1, 7), device)
        model3 = to_device(ResNet18(BasicBlock, [2,2,2,2], 7), device)
        evaluate(model1, validDataLoader)
        max_lr = 0.001
        grad_clip = 0.1
        weight_decay = 1e-4
        trainLog1 = fit(epoch, max_lr, model1, trainDataLoader, validDataLoader, weight_decay, grad_clip, torch.optim.Adam)
        trainLog2 = fit(epoch, max_lr, model2, trainDataLoader, validDataLoader, weight_decay, grad_clip, torch.optim.Adam)
        trainLog3 = fit(epoch, max_lr, model3, trainDataLoader, validDataLoader, weight_decay, grad_clip, torch.optim.Adam)
        torch.save(model1.state_dict(), type+'.pth')
        torch.save(model2.state_dict(), type+'.pth')
        torch.save(model3.state_dict(), type+'.pth')
        plot_losses_all(trainLog1, trainLog2, trainLog3)
        return
        
    print("Get the model type:" , type)
    print("Start model evaluation:")
    evaluate(model, validDataLoader)
    max_lr = 0.001
    grad_clip = 0.1
    weight_decay = 1e-4
    print("Start model fitting:")
    trainLog = fit(epoch, max_lr, model, trainDataLoader, validDataLoader, weight_decay, grad_clip, torch.optim.Adam)
    torch.save(model.state_dict(), type+'.pth')
    plot_losses(trainLog)
    plot_lrs(trainLog)
    
def testBias(t):
    data = np.load("./data/test.npz")
    model = None
    if t == "VGG":
        model = VGG(1, 7)
    elif t == "ResNet09":
        model = ResNet09(1, 7)
    else:
        model = ResNet18(BasicBlock, [2,2,2,2], 7)
    
    model.load_state_dict(torch.load(t + ".pth", map_location=get_default_device()))
    model.cuda()
    res = []

    valid_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    batch_num = 120

    for i in range(7):
        test_images = data["test_images"+str(i)]
        test_labels = data["test_labels"+str(i)]
        valid_data = Dataset(test_images, test_labels, valid_transform)
        validDataLoader = DataLoader(valid_data, batch_num*2, num_workers=4, pin_memory=True)
        device = get_default_device()
        validDataLoader = DeviceDataLoader(validDataLoader, device)
        result = evaluate(model, validDataLoader)
        print("Emotion [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            i, result['val_loss'], result['val_acc']))
        res.append(result['val_acc'])

    return res
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Type")
    parser.add_argument("--type", help="VGG or ResNet09 or ResNet18", required=True)
    args = parser.parse_args()
    main(args.type, 400, 30)
    testBias("ResNet18")
