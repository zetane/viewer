import cv2
import onnxruntime as ort
import argparse
import numpy as np
from box_utils import predict
import zetane as ztn

# Original script: ONNX Model Zoo
# https://github.com/onnx/models/tree/master/vision/body_analysis

# Face detection using UltraFace-320 onnx model
dir_path = os.path.dirname(os.path.realpath(__file__))
face_detector_onnx = os.path.join(dir_path, 'version-RFB-320.onnx')
face_detector = ort.InferenceSession(face_detector_onnx)

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

# face detection method
def faceDetector(orig_image, threshold = 0.7):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs

# Face age classification using VGG-16 onnx model
age_classifier_onnx = os.path.join(dir_path, 'vgg_ilsvrc_16_age_imdb_wiki.onnx')
age_classifier = ort.InferenceSession(age_classifier_onnx)

# age classification method
def ageClassifier(orig_image):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = age_classifier.get_inputs()[0].name
    ages = age_classifier.run(None, {input_name: image})
    age = round(sum(ages[0][0] * list(range(0, 101))), 1)
    return age



img_path = os.path.join(dir_path, 'pic.jpg')
color = (255, 255, 255)

orig_image = cv2.imread(img_path)
boxes, labels, probs = faceDetector(orig_image)

box = scale(boxes[0, :])
cropped = cropImage(orig_image, box)
age = ageClassifier(cropped)    
cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 8)
cv2.putText(orig_image, f'{age}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 6, color, 8, cv2.LINE_AA)
output_path =  os.path.join(dir_path, 'output.jpg')
cv2.imwrite(output_path, orig_image) 


zcontext = ztn.Context()
zcontext.launch()
zcontext.clear_universe()
zmodel = zcontext.model()
zimg_1 = zcontext.image()
zimg_2 = zcontext.image()

image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image = np.transpose(image, [2, 0, 1])
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)

zmodel.onnx(age_classifier_onnx).update()
zmodel.inputs(inputs = image).update()

### Sending images and texts to the engine
X = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
zimg_1.scale(.01,.01,.01).position(0,5,1).update(data= X/255)
ztxt_1 = zcontext.text("Input image:").position(-2,6.5,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()
    
orig_image = cv2.imread(img_path)
X2 = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
zimg_2.scale(.01,.01,.01).position(0,8,1).update(data= X2/255)
ztxt_2 = zcontext.text("Original image:").position(-2,14,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()
    
ztxt_3 = zcontext.text('Detected age: ' + f'{age}').position(-2,4,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()