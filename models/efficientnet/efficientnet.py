from PIL import Image
import numpy as np
import os
import sys
import zetane as ztn
import onnxruntime as ort
from tensorflow.keras.applications.vgg16 import decode_predictions
import onnx

# Original script: ONNX Model Zoo
# https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4

directory = os.path.dirname(__file__)

def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = img.resize((224, 224))
    img = np.asarray(img, dtype='float32')
    img = center_crop(img, output_height, output_width)
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img


# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

zcontext = ztn.Context().launch()
zonnx = zcontext.model()
zimg_1 = zcontext.image()


model = directory+r'/efficientnet-lite4-11.onnx'
session = ort.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

#### Data preparation and illustraion
path = directory+r'/cat.jpg'
X = Image.open(path)
X_ztn = X.resize((224, 224))
X_ztn = np.asarray(X_ztn)
X = pre_process_edgetpu(X, (224, 224, 3))
X = np.expand_dims(X, 0)

zimg_1.scale(.1,.1,.1).position(-4.5,4,1).update(data= X_ztn/255)
ztxt_1 = zcontext.text("Input image:").position(-6,5,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()

zonnx.visualize_inputs(False)
new_model_path = zonnx.onnx(model).update(inputs = X)


#### Inference
result = session.run([output_name], {input_name: X.astype(np.float32)})
outputs= np.reshape(result[0][0], (1, 1000))

ztxt_12 = zcontext.text(decode_predictions(outputs, top=5)[0][0][1]).position(-4,2,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_13 = zcontext.text(decode_predictions(outputs, top=5)[0][1][1]).position(-4,1.5,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_14 = zcontext.text(decode_predictions(outputs, top=5)[0][2][1]).position(-4,1,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_15 = zcontext.text(decode_predictions(outputs, top=5)[0][3][1]).position(-4,0.5,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_16 = zcontext.text(decode_predictions(outputs, top=5)[0][4][1]).position(-4,0,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_17 = zcontext.text("Top 5 outputs:").position(-6,2,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
