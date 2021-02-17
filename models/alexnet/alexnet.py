import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
import os
from tensorflow.keras.applications.vgg16 import decode_predictions
import sys
import zetane as ztn
sys.path.append('../')


# Launch Zetane
zcontext = ztn.Context()
zcontext.clear_universe()
zonnx = zcontext.model()
zimg_1 = zcontext.image()
zimg_2 = zcontext.image()

directory = os.path.dirname(__file__)

model = directory+'/alexnet.onnx'
width = 224
height = 224
filepath = os.getcwd()
resized_img_name = directory+'/background.jpg'
ort_session = ort.InferenceSession(model)
input_name = ort_session.get_inputs()[0].name
input_img_path = resized_img_name
img = Image.open(input_img_path)
X = np.asarray(img)
zimg_1.scale(.02,.02,.02).position(-11,-1,1).update(data=X/255)

img = img.resize((224, 224))
X = np.asarray(img)
zimg_2.scale(.02,.02,.02).position(-4.5,-1,1).update(data=X/255)

X = X / 255.
X = X.transpose(2, 0, 1)
X = X.reshape(1, 3, 224, 224)

ztxt_1 = zcontext.text("Original Image").position(-10,-2.5,1).font('roboto-mono').font_size(0.14).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_2 = zcontext.text("Scaling").position(-5,-2.5,1).font('roboto-mono').font_size(0.14).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_5 = zcontext.text("(1666, 2500, 3)").position(-10,-2,1).font('roboto-mono').font_size(0.14).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_6 = zcontext.text("(224, 224, 3)").position(-5,-2,1).font('roboto-mono').font_size(0.14).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_9 = zcontext.text("Pre-processing").position(-7,3,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()


zonnx.visualize_inputs(False)
new_model_path = zonnx.onnx(model).position(0,-5,1).update(inputs = X)

outputs = ort_session.run(None, {'data_0': X.astype(np.float32)})
print('Predicted', decode_predictions(outputs[0], top=10)[0])

ztxt_12 = zcontext.text(decode_predictions(outputs[0], top=5)[0][0][1]).position(-7,-4,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_13 = zcontext.text(decode_predictions(outputs[0], top=5)[0][1][1]).position(-7,-4.5,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_14 = zcontext.text(decode_predictions(outputs[0], top=5)[0][2][1]).position(-7,-5,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_15 = zcontext.text(decode_predictions(outputs[0], top=5)[0][3][1]).position(-7,-5.5,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_16 = zcontext.text(decode_predictions(outputs[0], top=5)[0][4][1]).position(-7,-6,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
ztxt_16 = zcontext.text("Top 5 outputs:").position(-11,-4,1).font('roboto-mono').font_size(0.17).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1))\
    .update()
