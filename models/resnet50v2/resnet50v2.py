from PIL import Image
import numpy as np
import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import zetane as ztn
import onnxruntime as ort
from tensorflow.keras.applications.vgg16 import decode_predictions
import onnx

# Original script: ONNX Model Zoo
# https://github.com/onnx/models/tree/master/vision/classification/resnet

directory = os.path.dirname(__file__)
def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


zcontext = ztn.Context().launch()
zonnx = zcontext.model()
zimg_1 = zcontext.image()


model = directory+r'/resnet50-v2-7.onnx'
session = ort.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

#### Data preparation and illustraion
path = directory+r'/car.jpg'
X = Image.open(path)
X = X.resize((224, 224))
X = np.asarray(X)

zimg_1.scale(.1,.1,.1).position(-8.5,34,1).update(data= X/255)
ztxt_1 = zcontext.text("Input image:").position(-12,35,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()

X = np.transpose(X, (2, 0, 1))
X = preprocess(X)
X = np.expand_dims(X, 0)


zonnx.visualize_inputs(False)
new_model_path = zonnx.onnx(model).update(inputs = X)


#### Inference
result = session.run([output_name], {input_name: X})
outputs= np.reshape(result[0][0], (1, 1000))

ztxt_12 = zcontext.text(decode_predictions(outputs, top=5)[0][0][1]).position(-8,32,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_13 = zcontext.text(decode_predictions(outputs, top=5)[0][1][1]).position(-8,31.5,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_14 = zcontext.text(decode_predictions(outputs, top=5)[0][2][1]).position(-8,31,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_15 = zcontext.text(decode_predictions(outputs, top=5)[0][3][1]).position(-8,30.5,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_16 = zcontext.text(decode_predictions(outputs, top=5)[0][4][1]).position(-8,30,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_17 = zcontext.text("Top 5 outputs:").position(-12,32,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
