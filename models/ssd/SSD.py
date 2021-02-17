from PIL import Image
import numpy as np
import onnxruntime as rt
import os
import sys
import zetane as ztn
import numpy as np
from PIL import Image
sys.path.append('../')
# TODO: Show the output images in zetane when the model renders properly in future

def preprocess(img_path):
    
    input_shape = (1, 3, 1200, 1200)
    img = Image.open(img_path)
    img = img.resize((1200, 1200), Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data
    
zcontext = ztn.Context()
zcontext.launch()
zcontext.clear_universe()
zmodel = zcontext.model()
zimg_input = zcontext.image()
zimg_output = zcontext.image()

directory = os.path.dirname(__file__)
model = directory+'/ssd-10.onnx'

img = Image.open(directory+'/demo.jpg')

ssd_image_data = preprocess(directory+'/demo.jpg')

session = rt.InferenceSession(model)
output_name_1 = session.get_outputs()[0].name
output_name_2 = session.get_outputs()[1].name
output_name_3 = session.get_outputs()[2].name

input_name = session.get_inputs()[0].name

boxes,labels, scores = session.run([output_name_1, output_name_2, output_name_3], {input_name: ssd_image_data})

zmodel.visualize_inputs(False)
zmodel.onnx(model).use_hierarchical_layout(False).update(inputs = ssd_image_data)
