"""
ImageNet Explain
===========================

"""

import os
import sys
os.environ['TF_KERAS'] = '1'

import onnx
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import zetane.context as ztn
from zetane.XAI_dashboard import XAIDashboard

from tensorflow.keras.applications.vgg16 import VGG16

class_dict = dict()
with open("imagenet_classes.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        class_dict[i] = line

example_list = (('./input_images/snake.jpg', 56),
                    ('./input_images/cat_dog.png', 243),
                    ('./input_images/spider.png', 72))
example_index = 0
img_path = example_list[example_index][0]
label_class = example_list[example_index][1]

# Create the model that will be used by different explainability modules and classes
model = VGG16(weights='imagenet')
model.compile(loss='mean_squared_error', optimizer='adam')

zcontext = ztn.Context()
zcontext.clear_universe()

explain_template = XAIDashboard(model, zcontext)
explain_template.explain_keras(img_path, None, label_class, class_dict, algorithms=None)

zcontext.disconnect()
