import matplotlib.pyplot as plt
import numpy as np
import torch, os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from util import *


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

class ResNet(Base):
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
    print(model)
    outputs = [model.validating(batch) for batch in val_loader]
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
        
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
    
    
def main(type):
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
    torch.manual_seed(209)
    batch_num = 120
    trainDataLoader = DataLoader(train_data, batch_num, shuffle=True, num_workers=4, pin_memory=True)
    validDataLoader = DataLoader(valid_data, batch_num*2, num_workers=4, pin_memory=True)
    print("TrainDataLoader and validDataLoader initialization finished")
    device = get_default_device()
    trainDataLoader = DeviceDataLoader(trainDataLoader, device)
    validDataLoader = DeviceDataLoader(validDataLoader, device)
    if type=="VGG":
        model = to_device(VGG(1, 7), device)
    else:
        model = to_device(ResNet(1, 7), device)
    print("Get the model type:" , type)
    print("Start model evaluation:")
    evaluate(model, validDataLoader)
    max_lr = 0.001
    grad_clip = 0.1
    weight_decay = 1e-4
    print("Start model fitting:")
    trainLog = fit(30, max_lr, model, trainDataLoader, validDataLoader, weight_decay, grad_clip, torch.optim.Adam)
    torch.save(model.state_dict(), type+'.pth')
    plot_losses(trainLog)
    plot_lrs(trainLog)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Type")
    parser.add_argument("--type", help="VGG or ResNet", required=True)
    args = parser.parse_args()
    main(args.type)
