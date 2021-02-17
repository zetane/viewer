"""
ImageNet Explain
===========================

"""

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np

import zetane.context as ztn

from zetane.XAI_dashboard import XAIDashboard

class_dict = dict()
with open("imagenet_classes.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        class_dict[i] = line

# mean and std list for channels (Imagenet)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

example_list = (('./input_images/snake.jpg', 56),
                    ('./input_images/cat_dog.png', 243),
                    ('./input_images/spider.png', 72))
example_index = 0
img_path = example_list[example_index][0]
label_class = example_list[example_index][1]
original_image = Image.open(img_path).convert('RGB')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(pretrained=True)
model = model.to(device)

zcontext = ztn.Context()
zcontext.clear_universe()

explain_template = XAIDashboard(model, zcontext)
explain_template.explain_torch(img_path, None, label_class, class_dict, algorithms=None, mean=mean, std=std)

zcontext.disconnect()
