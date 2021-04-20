import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch, os
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN


def bouding_boxes(path, result_list):
    img = pyplot.imread(path)
    pyplot.imshow(img)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            ax.add_patch(Circle(value, radius=1, color='red'))
    pyplot.show()
    
    
def headPoseEstimation(path):
    img = pyplot.imread(path)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    right_eyes = []
    left_eyes = []
    for face in faces:
        right_eyes.append(face['keypoints']['right_eye'])
        left_eyes.append(face['keypoints']['left_eye'])
    angles = []
    for i in range(len(right_eyes)):
        angles.append((right_eyes[i][1] - left_eyes[i][1]) / (right_eyes[i][0] - left_eyes[i][0]))
    bouding_boxes(path, faces)
    output = save_faces(path, faces)
    return output, faces
    
    
def save_faces(path, result_list):
    data = pyplot.imread(path)
    h, w = data.shape[0], data.shape[1]
    output = np.empty((len(result_list), 48*48))
    for i in range(len(result_list)):
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        x1 = max(0, x1)
        x2 = min(x2, w)
        y1 = max(0, y1)
        y2 = min(y2, h)
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        img = data[y1:y2, x1:x2]
        res = cv2.resize(img, dsize = (48, 48), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        output[i] = gray.reshape((1, 48*48))
        pyplot.imshow(res)
    pyplot.show()
    return output
    
    
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
    
    
def accuracy(outputs, labels):
    predictions = torch.max(outputs, dim=1)[1]
    return torch.tensor(torch.sum(predictions==labels).item()/len(predictions))
    
    
def get_default_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    else: return torch.device('cpu')
    
    
def to_device(data, device=get_default_device()):
    if isinstance(data, (list,tuple)): return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def plot_losses_all(history1, history2, history3):
    train_losses1 = [x.get('train_loss') for x in history1]
    train_losses2 = [x.get('train_loss') for x in history2]
    train_losses3 = [x.get('train_loss') for x in history3]
    val_losses1 = [x['val_loss'] for x in history1]
    val_losses2 = [x['val_loss'] for x in history2]
    val_losses3 = [x['val_loss'] for x in history3]
    plt.plot(train_losses1, '-b', label="VGG TL")
    plt.plot(train_losses2, '-p', label="ResNet09 TL")
    plt.plot(train_losses3, '-g', label="ResNet18 TL")
    plt.plot(val_losses1, '-k', label="VGG VL")
    plt.plot(val_losses2, '-m', label="ResNet09 VL")
    plt.plot(val_losses3, '-c', label="ResNet18 VL")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss vs. No. of epochs')
    plt.savefig("res.png")

