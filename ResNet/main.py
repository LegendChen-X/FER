import cv2, torch
import numpy as np
from model import *
from headpose import *
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
    model = Model(1, 7)
    softmax = torch.nn.Softmax(dim=1)
    model.load_state_dict(torch.load('9.pth', map_location=get_default_device()))
    out, faces = headPoseEstimation("test.PNG")
    image = cv2.imread("test.PNG")
    for i, face in zip(out, faces):
        img = torch.from_numpy(i.reshape((48, 48)))
        img = imgTensor(img)
        prediction = predict(img)
        print(prediction['label'], prediction['probability'])
#cv2.imshow("face_to_emoji", image)
#cv2.imwrite("face_to_emoji.jpg", image)
#cv2.waitKey(0)
