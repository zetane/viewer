import numpy as np
import string
import onnxruntime as rt

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import zetane as ztn


import urllib.request

# urllib is a built-in Python library to download files from URLs
directory = os.path.dirname(__file__)
output_classes_url = "https://raw.githubusercontent.com/qqwweee/keras-yolo3/master/model_data/coco_classes.txt"
urllib.request.urlretrieve(output_classes_url, filename=directory+"/coco_classes.txt")
classes = [line.rstrip('\n') for line in open(directory+'/coco_classes.txt')]

dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, 'horses.jpg') # sample file, change with your own image
input_path = os.path.join(dir_path, 'yolov3_inputs.npz')

tinyYOLOv3 = False

if tinyYOLOv3:
    # download from https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny_yolov3
    onnx_path = os.path.join(dir_path, 'yolov3-tiny.onnx')
else:
    # download from https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3
    onnx_path = os.path.join(dir_path, 'yolov3.onnx')

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    #resize image with unchanged aspect ratio using padding
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data


image = Image.open(img_path)
# input
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)

# Launch Zetane
zcontext = ztn.Context()
zcontext.clear_universe()
np.savez(input_path, image_data, image_size)

# Preprocess image
# Load onnx file and run inference

session = rt.InferenceSession(onnx_path)

if tinyYOLOv3:
    output_names = ["yolonms_layer_1", "yolonms_layer_1:1", "yolonms_layer_1:2"]
else:
    output_names = ["yolonms_layer_1/ExpandDims_1:0", "yolonms_layer_1/ExpandDims_3:0", "yolonms_layer_1/concat_2:0"]

boxes, scores, indices = session.run(output_names, {"input_1": image_data, "image_shape":image_size})

print('Boxes:', boxes.shape)
print('Scores:', scores.shape)
print('Indices:', indices.shape)


zimg = zcontext.image()
zonnx = zcontext.model().onnx(onnx_path).inputs(input_path)
zonnx.visualize_inputs(False)
zonnx.update()
print('update')

def postprocess(boxes, scores, indices):
    objects_identified = indices.shape[0]
    out_boxes, out_scores, out_classes = [], [], []
    if objects_identified > 0:
        for idx_ in indices:
            out_classes.append(classes[idx_[1]])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])
        print(objects_identified, "objects identified in source image.")
    else:
        print("No objects identified in source image.")
    return out_boxes, out_scores, out_classes, objects_identified

if tinyYOLOv3:
    out_boxes, out_scores, out_classes, objects_identified = postprocess(boxes, scores, indices[0])
else:
    out_boxes, out_scores, out_classes, objects_identified = postprocess(boxes, scores, indices)


def display_objdetect_image(image, out_boxes, out_classes, \
                            image_name='sample', objects_identified=None, save=True):
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(image)
    if objects_identified == None:
        objects_identified = len(out_boxes)

    for i in range(objects_identified):
        y1, x1, y2, x2 = out_boxes[i]
        class_pred = out_classes[i]
        color = 'blue'
        box_h = (y2 - y1)
        box_w = (x2 - x1)
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1, s=class_pred, color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})

    plt.axis('off')
    # save image
    image_name = os.path.join(dir_path, image_name + "-det.jpg")
    plt.savefig(image_name, bbox_inches='tight', pad_inches=0.0)

    zimg.position(0, -7, 0).rotation(0,-3.1416 / 2,0).update(filepath=image_name)


display_objdetect_image(image, out_boxes, out_classes, "horse")

zcontext.disconnect()