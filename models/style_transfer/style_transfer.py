import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os import walk
import zetane.context as ztn
from PIL import Image
import numpy as np
import onnxruntime as rt
import pathlib
import time
from pynput.keyboard import Key, Listener



class Style_transfer_ztn():
    def __init__(self):
        pass 

zcontext = ztn.Context()
zcontext.clear_universe()
mosaic = zcontext.model()
actual_img = zcontext.image()
output_img = zcontext.image()
picture_number = 1
running = True
change = True
change_model = True
directory = os.path.dirname(__file__)
model_name = directory + "/rain-princess-9.onnx"

# loading input and resize if needed
images = []
for p in pathlib.Path(directory +'/pics').iterdir():
    if p.is_file():
        images.append(Image.open(p).convert('RGB'))
images = list(map(lambda image: image.resize((224, 224), Image.NEAREST), images))

max = len(images)

def on_press(key):
    global picture_number
    global change
    global model_name
    global change_model
    
    if key == Key.left:
        picture_number -= 1
        change = True
        print("left")
    elif key == Key.right:
        picture_number += 1
        change = True
        print("right")
    else:
        if hasattr(key, 'char'):
            if key.char == "1":
                model_name = "rain-princess-9.onnx"
                change_model = True
            elif key.char == "2":
                model_name = "mosaic-8.onnx"
                change_model = True
            elif key.char == "3":
                model_name = "udnie-8.onnx"
                change_model = True
            elif key.char == "4":
                model_name = "pointilism-8.onnx"
                change_model = True
            elif key.char == "5":
                model_name = "candy-8.onnx"
                change_model = True

    if picture_number >= max - 1:
        picture_number = max - 1
    elif picture_number <= 0:
        picture_number = 0


def on_release(key):
    if key == Key.esc:
        global running
        running = False
        return False


listener = Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


# Preprocess image
def preprocess(image, i):
    x = np.array(image).astype('float32')
    # actual_img.position(x=-10.0, y=(2.5 * 1)).scale(0.05, 0.05).update(data=x/255.0)
    x_t = np.transpose(x, [2, 0, 1])
    x_t = np.expand_dims(x_t, axis=0)
    return x, x_t

def onnx(image, i):
    session = rt.InferenceSession(model_name)
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name
    result = session.run([output_name], {input_name: x_t})[0][0]
    # postprocess
    result = np.clip(result, 0, 255)
    result = result.transpose(1, 2, 0).astype("uint8")
    return result

while running:
    if change or change_model:
        if change_model:
            zcontext.clear_universe()
            change_model = False
        print(picture_number)
        x, x_t = preprocess(images[picture_number], picture_number)
        mosaic.onnx(model_name).inputs(x_t).update()


        res = onnx(images[picture_number], picture_number)
        output_img.position(x=0.0, y=1.0).scale(0.3, 0.3).update(data=res / 255)
        actual_img.position(x=-7.0, y=1.0).scale(0.3, 0.3).update(data=x / 255.0)
        change = False