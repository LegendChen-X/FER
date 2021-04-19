import cv2, torch
import numpy as np
import argparse
from model import *
from util import *
from matplotlib import pyplot


def imgTensor(img):
    img_transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
    return img_transform(img)

def predict(img, model):
    model_out = model(imgTensor(img)[None])
    softmax = torch.nn.Softmax(dim=1)
    soft_out = softmax(model_out)
    probability = torch.max(soft_out).item()
    label = classes[torch.argmax(soft_out).item()]
    label_index = torch.argmax(soft_out).item()
    return {'label': label, 'probability': probability, 'index': label_index}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image")
    parser.add_argument("--image", help="path of image", required=True)
    parser.add_argument("--type", help="VGG or ResNet", required=True)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if args.type=="VGG":
        model = VGG(1, 7)
    else:
        model = ResNet(1, 7)
    print("Get the model type:", args.type)
    model.load_state_dict(torch.load(args.type+".pth", map_location=get_default_device()))
    outputs, faces = headPoseEstimation(args.image)
    for output, face in zip(outputs, faces):
        image = torch.from_numpy(output.reshape((48, 48)))
        reslut = predict(image, model)
        print("The predicted expression is："+ reslut['label'] + " with probability：%f" % reslut['probability'])