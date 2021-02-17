import os 
os.environ['TF_KERAS'] = '1'
import keras2onnx
import onnx
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import zetane as ztn 

model = load_model('Xray_80_whole_f2.h5')
zcontext = ztn.Context()
zcontext.clear_universe()
zmodel = zcontext.model()
zimg_1 = zcontext.image()
zimg_2 = zcontext.image()
zimg_3 = zcontext.image()

filepath = os.getcwd()
orig_img = filepath + '\\' +'JPCLN002_img.PNG'
img = Image.open(orig_img)
img = img.resize((256, 256))
X = np.asarray(img)
zimg_1.scale(.03,.03,.03).position(-8,3,1).update(data=X/255)
ztxt_1 = zcontext.text("Original image").position(-8,2.5,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()

X = np.expand_dims(img,2)
X = np.expand_dims(X,0)
y = model.predict(X)
y[y> 0.5] = 1
y[y<= 0.5] = 0
zonnx = zcontext.model().keras(model).update(inputs=X)
zimg_2.scale(.03,.03,.03).position(-6,3,1).update(data= y[0,:,:,0])
ztxt_2 = zcontext.text("Predicted mask").position(-6,2.5,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()

orig_mask = filepath + '\\' +'JPCLN002_mask.PNG'
mask = Image.open(orig_mask)
mask = mask.resize((256, 256))
X = np.asarray(mask)
zimg_3.scale(.03,.03,.03).position(-4,3,1).update(data=X/255)
ztxt_3 = zcontext.text("Ground truth").position(-4,2.5,1).font('roboto-mono').font_size(0.12).billboard(True) \
    .color((1, 1, 1)).highlight((0.5, 0.5, 1)).update()







dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filename = os.path.join(dir, 'unet')
if not os.path.exists(filename + '/ztn'): os.makedirs(filename + '/ztn')
zcontext.save(filename + '/ztn/unet.ztn')
zcontext.clear_universe()
