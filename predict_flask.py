#Import Libraries
import numpy as np

''' IMPORTANT
    Only one of the following imports must be uncommented.
    Docker: 
        import tflite_runtime.interpreter as tflite
        # import tensorflow.lite as tflite
    Local testing: 
        #import tflite_runtime.interpreter as tflite
        import tensorflow.lite as tflite    
'''
#import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite

from io import BytesIO
from urllib import request
import urllib
from PIL import Image

from flask import Flask 
from flask import request
from flask import jsonify

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


app = Flask('classify')
 


@app.route('/predict', methods=['POST'])
def predict():
    
    client = request.get_json()
 
    img = download_image(client.get('url'))

    img_resized = prepare_image(img, (224, 224))
 
    X = preprocess_image(img_resized)
 
    pred = prediction(X)
 
    if (pred[0][0]>0.5):
        return  jsonify({ "Prediction": "normal",
                          "Predict Probability": str(pred[0][0])
                })
    else:
        return jsonify({ "Prediction": "fracture",
                         "Predict Probability": str(pred[0][0])
                  })
        
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)