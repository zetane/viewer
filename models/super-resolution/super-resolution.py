"""
***

# **Super-resolution**

![](super-resolution/screenshot.PNG)

### Description

This script used for super resolution of images.
<br /><br />
<a href="super-resolution/ztn/super-resolution.ztn"><button type="button" name="button" style="color:#000; background-color:#FFF887; border-radius:10px; width: 220px; height: 75px;margin-right:auto;font-size:20px">Download<br/> Snapshot</button></a>

<br /><br />

"""
# Original script: ONNX Model Zoo, Super Resolution
# https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016

from PIL import Image
import numpy as np
import onnxruntime as rt
import os
import sys
sys.path.append('../')
import zetane as ztn

# Launch Zetane



zcontext = ztn.Context()
zcontext.clear_universe()
zonnx = zcontext.model()



zimg_input = zcontext.image()
zimg_output = zcontext.image()
ztext_input = zcontext.text("Input (224 X 224)")
ztext_output = zcontext.text("Output (672 X 672)")

directory = os.path.dirname(__file__)
# provide the name of the model and width and height of the images for the variables
model = directory+'/super_resolution.onnx'
print(model)
width = 224
height = 224

# Load and downscale image
input_img_path = directory+'/test.jpg'
orig_img = Image.open(input_img_path)
downscaled_img = orig_img.resize((width, height))
downscaled_img.save(directory+"/test_downscaled.jpg")

# Preprocess image for the model
img_ycbcr = downscaled_img.convert('YCbCr')
img_y_0, img_cb, img_cr = img_ycbcr.split()
img_ndarray = np.asarray(img_y_0)
img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
img_5 = img_4.astype(np.float32) / 255.0

# Load onnx file and run inference
session = rt.InferenceSession(model)
output_name = session.get_outputs()[0].name
input_name = session.get_inputs()[0].name
result = session.run([output_name], {input_name: img_5})
img_out_y = result[0]
print(img_out_y.shape)

# Postprocess
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
# get the output image following the post-processing step from the PyTorch implementation
final_img = Image.merge(
                "YCbCr", [
                                img_out_y,
                                img_cb.resize(img_out_y.size, Image.BICUBIC),
                                img_cr.resize(img_out_y.size, Image.BICUBIC),
                ]).convert("RGB")

# Display in the Zetane engine
# Downscaled image
image_for_zetane = np.asarray(downscaled_img)
image_for_zetane = image_for_zetane.astype(np.float32) / 255.0
zimg_input.position(-4.5, 1, 0).scale(0.1, 0.1, 0.1).update(data=image_for_zetane)

# Model architecture and tensors
zonnx.onnx(model).update(inputs = img_5)

# Model output super resolution image
final_img = np.asarray(final_img)
final_img = final_img.astype(np.float32) / 255.0
zimg_output.position(-1.5, 1, 0).scale(0.0335,0.0335,0.0335).update(data=final_img)
# ztext_output.position(0.41,3.48,0).font_size(0.12).update()

ztext_input.position(-4.37, 3.48, 0).font_size(0.12).update()
ztext_output.position(-1.35, 3.48, 0).font_size(0.12).update()
