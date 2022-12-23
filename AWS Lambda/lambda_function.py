#Import Libraries
import numpy as np
import tflite_runtime.interpreter as tflite
import urllib
from io import BytesIO
from urllib import request
import json
from PIL import Image


## Import tflite model 
interpreter = tflite.Interpreter(model_path='cerv_fracture_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0].get('index')
output_index = interpreter.get_output_details()[0].get('index')
 

## Image Download, resize, preprocess, predict and lambda_handler functions' definitions
def download_image(url):
    with urllib.request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_image(img):

    x = np.asarray(img)
    X = np.array([x/255.], dtype="float32")
    return X


def prediction(X):
     
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_index)

    return pred


def lambda_handler(event, context=None):

    url = event['url']
    
    img = download_image(url)

    img_resized = prepare_image(img, (224, 224))

    X = preprocess_image(img_resized)

    pred = prediction(X)

    if (pred[0][0]>0.5):
        return  json.dumps({ "Prediction": "normal",
                  "Predict Probability": str(pred[0][0])
                })
    else:
        return  json.dumps({  "Prediction": "fracture",
                  "Predict Probability": str(pred[0][0])
                })