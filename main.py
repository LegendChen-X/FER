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


def predictFull(img, model):
    model_out = model(imgTensor(img)[None])
    softmax = torch.nn.Softmax(dim=1)
    soft_out = softmax(model_out)
    return soft_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image")
    parser.add_argument("--image", help="path of image", required=True)
    parser.add_argument("--type", help="VGG or ResNet09 or ResNet18", required=True)
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if args.type == "VGG":
        model = VGG(1, 7)
    elif args.type == "ResNet09":
        model = ResNet09(1, 7)
    else:
        model = ResNet18(BasicBlock, [2,2,2,2], 7)
    print("Get the model type:", args.type)
    model.load_state_dict(torch.load(args.type + ".pth", map_location=get_default_device()))
    outputs, faces = headPoseEstimation(args.image)
    for output, face in zip(outputs, faces):
        image = torch.from_numpy(output.reshape((48, 48)))
        reslut = predict(image, model)
        print("The predicted expression is：" + reslut['label'] + " with probability：%f" % reslut['probability'])

        full_reslut = predictFull(image, model)
        plt.figure(figsize=(12, 5))
        axes = plt.subplot(1, 2, 1)
        pic = plt.imread(args.image)
        plt.imshow(pic)
        plt.xlabel('Image', fontsize=12)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.tight_layout()

        classes = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']
        plt.subplots_adjust(bottom=0.2, top=0.8, wspace=0.3)
        plt.subplot(1, 2, 2)
        colors = ['red', 'orange', 'gold', 'yellow', 'greenyellow', 'green', 'royalblue']
        plt.title("Facial Expression probabilitis", fontsize=17)
        plt.xlabel("Facial Expression", fontsize=15)
        plt.ylabel("Probability", fontsize=15)
        ind = 0.1 + 0.5 * np.arange(7)
        plt.xticks(ind, classes, rotation=30, fontsize=12)
        for i in range(7):
            plt.bar(0.1 + 0.5 * i, full_reslut[0][i].item(), 0.3, color=colors[i])

        plt.savefig(os.path.join('result.png'))
        plt.close()
