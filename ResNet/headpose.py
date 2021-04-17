from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np


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
