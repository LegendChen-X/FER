import matplotlib.pyplot as plt
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import pandas as pd


classes = {
    0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happy', 4: 'Sad', 5: 'Anger', 6: 'Neutral'
}

def accuracy(outputs, labels):
    predictions = torch.max(outputs, dim=1)[1]
    return torch.tensor(torch.sum(predictions==labels).item()/len(predictions))
    
    
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')

class Base(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

def conv_block(in_chnl, out_chnl, pool=False, padding=1):
    layers = [nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=padding), nn.BatchNorm2d(out_chnl), nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Model(Base):
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

def get_default_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    else: return torch.device('cpu')
    
def to_device(data, device=get_default_device()):
    if isinstance(data, (list,tuple)): return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class Dataset(Dataset):

    def __init__(self, images, labels, transforms):
        self.X = images
        self.y = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        data = np.asarray(data).astype(np.uint8).reshape(48,48,1)
        data = self.transforms(data)
        label = self.y[i]
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
# This function will evaluate the model and give back the val acc and loss
    model.eval()
    print(model)
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
    
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
        model.epoch_end(epoch, result)
        history.append(result)
    return history
    
def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    
def main():
    print("Get data successfully")
    npzfile = np.load("./../data/raf_db.npz")
    npzfile1 = np.load("./../data/toronto_face.npz")

    train_images = npzfile["inputs_train"]
    train_labels = np.argmax(npzfile["target_train"], axis=1)
    test_images = npzfile["inputs_valid"]
    test_labels = np.argmax(npzfile["target_valid"], axis=1)

    train_images1 = npzfile1["inputs_train"]
    train_labels1 = npzfile1["target_train"]
    test_images1 = npzfile1["inputs_valid"]
    test_labels1 = npzfile1["target_valid"]


    for i in range(len(train_labels1)):
        if train_labels1[i] == 0:
            train_labels1[i] = 5
        elif train_labels1[i] == 1:
            train_labels1[i] = 2
        elif train_labels1[i] == 2:
            train_labels1[i] = 1
        elif train_labels1[i] == 3:
            train_labels1[i] = 3
        elif train_labels1[i] == 4:
            train_labels1[i] = 4
        elif train_labels1[i] == 5:
            train_labels1[i] = 0
        elif train_labels1[i] == 6:
            train_labels1[i] = 6
        else:
            print("wrong train label",train_labels1[i])
    
    for i in range(len(test_labels1)):
        if test_labels1[i] == 0:
            test_labels1[i] = 5
        elif test_labels1[i] == 1:
            test_labels1[i] = 2
        elif test_labels1[i] == 2:
            test_labels1[i] = 1
        elif test_labels1[i] == 3:
            test_labels1[i] = 3
        elif test_labels1[i] == 4:
            test_labels1[i] = 4
        elif test_labels1[i] == 5:
            test_labels1[i] = 0
        elif test_labels1[i] == 6:
            test_labels1[i] = 6
        else:
            print("wrong test label",test_labels1[i])

    train_images = np.concatenate((train_images, train_images1))
    train_labels = np.concatenate((train_labels, train_labels1))
    test_images = np.concatenate((test_images, test_images1))
    test_labels = np.concatenate((test_labels, test_labels1))

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

    print("Initialize train data successfully")
    train_data = Dataset(train_images, train_labels, train_transform)
    valid_data = Dataset(test_images, test_labels, valid_transform)
    torch.manual_seed(33)
    batch_num = 120
    print("Get trainDataLoader successfully")
    trainDataLoader = DataLoader(train_data, batch_num, shuffle=True, num_workers=4, pin_memory=True)
    validDataLoader = DataLoader(valid_data, batch_num*2, num_workers=4, pin_memory=True)
    device = get_default_device()
    trainDataLoader = DeviceDataLoader(trainDataLoader, device)
    validDataLoader = DeviceDataLoader(validDataLoader, device)
    print("Get model successfully")
    model = to_device(Model(1, 7), device)
    print("Begin evalute")
    evaluate(model, validDataLoader)
    max_lr = 0.001
    grad_clip = 0.1
    weight_decay = 1e-4
    print("Begin fit")
    trainLog = fit(50, max_lr, model, trainDataLoader, validDataLoader, weight_decay, grad_clip, torch.optim.Adam)
    torch.save(model.state_dict(), '9.pth')
    plot_losses(trainLog)
    plt.figure()
    plot_lrs(trainLog)
    
if __name__ == "__main__":
    main()
