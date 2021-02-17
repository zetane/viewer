import cv2
import onnxruntime as ort
import argparse
import numpy as np
from box_utils import predict
import zetane as ztn
import os
import sys

# Original script: ONNX Model Zoo
# https://github.com/onnx/models/tree/master/vision/body_analysis

# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)
    
    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num

# pre-processing
def pre_processing(orig_image): 
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image

# face detection method
def faceDetector(orig_image, threshold = 0.7):
    image = pre_processing(orig_image)
    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

# Face detection using UltraFace-320 onnx model
dir_path = os.path.dirname(os.path.realpath(__file__))
model = os.path.join(dir_path, 'version-RFB-320.onnx')
face_detector = ort.InferenceSession(model)


zcontext = ztn.Context()
zcontext.launch()
zcontext.clear_universe()
zmodel = zcontext.model()
zimg_1 = zcontext.image()
zimg_2 = zcontext.image()


img_path = os.path.join(dir_path, 'pic.jpg')
color = (255, 128, 0)
orig_image = cv2.imread(img_path)

### Sending the input image to the engine
X = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
X = cv2.resize(X, (240,320))
zimg_1.scale(.1,.1,.1).position(-4.5,69,1).update(data= X/255)
ztxt_1 = zcontext.text("Input image:").position(-6,71,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()

X_ztn = pre_processing(orig_image)
boxes, labels, probs = faceDetector(orig_image)

box = scale(boxes[0, :])
cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 15)
output_path =  os.path.join(dir_path, 'output.jpg')
cv2.imwrite(output_path, orig_image) 


### Sending the output image to the engine
y = cv2.imread(output_path)
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
y = cv2.resize(y, (240,320))
zimg_2.scale(.1,.1,.1).position(-4.5,65,1).update(data= y/255)
ztxt_2 = zcontext.text("Output:").position(-6,67,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()

### sending the model to the engine
zmodel.onnx(model).inputs(inputs = X_ztn).update()
