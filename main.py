import cv2, torch
import numpy as np
import argparse
from model import *
from util import *
from matplotlib import pyplot


def imgTensor(x):
    transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    return transform(x)


def predict(x):
    out = model(imgTensor(img)[None])
    scaled = softmax(out)
    prob = torch.max(scaled).item()
    label = classes[torch.argmax(scaled).item()]
    label_num = torch.argmax(scaled).item()
    return {'label': label, 'probability': prob, 'index': label_num}
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image")
    parser.add_argument("--image", help="path of image", required=True)
    parser.add_argument("--type", help="VGG or ResNet", required=True)
    args = parser.parse_args()
    if args.type=="VGG":
        model = VGG(1, 7)
    else:
        model = ResNet(1, 7)
    softmax = torch.nn.Softmax(dim=1)
    model.load_state_dict(torch.load(args.type+".pth", map_location=get_default_device()))
    out, faces = headPoseEstimation(args.image)
    image = cv2.imread(args.image)
    for i, face in zip(out, faces):
        img = torch.from_numpy(i.reshape((48, 48)))
        img = imgTensor(img)
        prediction = predict(img)
        print(prediction['label'], prediction['probability'])
