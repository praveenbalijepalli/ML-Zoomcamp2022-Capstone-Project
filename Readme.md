# Project: Cervical Spinal Fracture Classification from C.T. scans
``` 
Artificial Intelligence  
Domain             : Computer Vision
Sub-Domain         : Deep Learning, Image Classification
Architectures      : Deep CNNs, Inceptionv3, Resnet50V2, Xception
Application        : Medical Image Classification
```

## Problem Statement 
The fractures to cervical region usually result from high energy trauma like automobile crashes and falls. In elderly people, a fall of the charir or falling on the ground can cause a cervical fracture. Cervical fracture is the fracture of any of the seven cervical vertebrae in the neck. Since they support the head and connect it to the shoulders and the body, immediate response to an injury to it paramount as it can have serious consequences. Injury to the vertebra can lead temporary or permanent paralysis from the neck down and in some cases even leading to death. So, a physician will usually need support from radiographic studies such as MRI or CT scans to determine the extent of the injuries. This project is an endeavour to use AI to assist a physician to determine if a CT scan image shows a "fracture" in a vertbrae or if it is "normal".



## Dataset Details
```
Dataset Description: Contains images of Fractured and Normal Cervical CT scans in their respective folders in Train and Test Folders
Dataset Name       : Spine Fracture Prediction from C.T. Dataset
Dataset Link       : https://www.kaggle.com/datasets/vuppalaadithyasairam/spine-fracture-prediction-from-xrays/code
Dataset Size       : 311 MB 
Number of Classes  : 2 (Fracture, Normal)
Number of Images   : Training Images - 3800 (Training  - 3040 and Validation - 760 Images
                     Testing Images - 400
```


## Parameters for Training
```
For Pre-trained Models:
Model Architecture      : Input, Pre-trained base models(Inceptionv3, Resnet50V2, Xception with imagenet weights), GlobalAveragePooling2D, Dense - Output layer

For Custom Deep CNN: 
Architecture            : CNN2D - 2 layers, Dropout, Batch Normalization, MaxPool2D, Flatten, Dense - 2 layers, Dense - Output layer
Regularization          : Dropout (0.2)
 
Optimizers              : SGD - Stochastic Gradient Descent
Loss Function           : binary_crossentropy
Batch Size              : 16
Number of Epochs        : 20
```

## Classification Metrics of the Final Model
```
Final Model             : https://github.com/praveenbalijepalli/mlzoomcamp-capstone-1/blob/main/cerv_fracture_model.h5
Final TFlite Model      : https://github.com/praveenbalijepalli/mlzoomcamp-capstone-1/blob/main/cerv_fracture_model.tflite
Train, Val and Test     
Accuracy Score          : Train  -     , Val -    , Test - 
Loss                    : Train  -     , Val -    , Test -   
Precision               : Train  -     , Val -    , Test -
Recall                  : Train  -     , Val -    , Test -
```

## Sample Input and Output

In the predict_test.py file, replace the data variable with the following:
1. Sample Input:
   ```
   data = {'url':'https://raw.githubusercontent.com/praveenbalijepalli/mlzoomcamp-capstone-1/main/sample%20images%20for%20testing/fracture.png'}
   ```

   Sample Output:
   ```
   {'Predict Probability': '0.06067303', 'Prediction': 'fracture'}
   ```
   
   
 2. Sample Input:
    ```
    data = {'url':'https://raw.githubusercontent.com/praveenbalijepalli/mlzoomcamp-capstone-1/main/sample%20images%20for%20testing/normal.png'}
    ```

    Sample Output:
    ```
    {'Predict Probability': '0.7252521', 'Prediction': 'normal'}
    ```
 
## Tools / Libraries
```
Languages               : Python
Tools/IDE               : Anaconda
Libraries               : Keras, TensorFlow
Virtual Environment     : pipenv
```

## Scripts
```
Train Script            : https://github.com/praveenbalijepalli/mlzoomcamp-capstone-1/blob/main/train.py
Keras to TFlite Script  : https://github.com/praveenbalijepalli/mlzoomcamp-capstone-1/blob/main/convert_keras_to_tflite.py
Predict Script          : https://github.com/praveenbalijepalli/mlzoomcamp-capstone-1/blob/main/predict_flask.py
Predict Test Script     : https://github.com/praveenbalijepalli/mlzoomcamp-capstone-1/blob/main/predict_test.py
```

## Run the Model as is  
Steps to run the scripts/notebooks as is:

1. Clone the repo by running the following command:
   ```
   git clone https://github.com/praveenbalijepalli/mlzoomcamp-capstone-1.git
   ```
2. Open a terminal or command prompt and change directory to the folder where this project is cloned.

3. Run the following command to activate the virtual environment for the project:
   ```
   pipenv shell
   ```

   In case, pipenv is not installed in your system, to install pipenv and to activate the virtual environment for the project, type the following commands:
   ```
   pip install pipenv 
   pipenv shell (in the project folder)
   ``` 
4.  To install the files and dependencies related to the project, run the following in the folder containing Pipfile/Pipfile.lock
    ```
    pipenv install
    ```
5.  To run the scripts do the following:

    a. To train the the model and save it using train.py script, run the following command in the terminal/prompt.
       ```
       python train.py (To train and save the model)
       ```
       
    b. Run predict_flask.py using python in a terminal/prompt.
       ```
       python predict_flask.py (To start the prediction service)
       ```
       
    c. Open another terminal/prompt and run predict_test.py.
       ```
       python predict_test.py (To test the prediction service)
       ```


## Model as a web service 

### Using Waitress  
   
   1. Follow the steps mentioned above from 1 to 4, if you haven't already completed them.
   
   2. To run the prediction service offered by predict_flask.py using waitress, type the following command
      ```
      waitress-serve --listen=0.0.0.0:9696 predict_flask:app (This will keep the running the prediction service)
      ```
      
   3. Open another terminal/prompt and run predict_test.py
      ```
      python predict_test.py (To test the prediction service)
      ``` 
      
 ### Using Docker 
 
   1. Clone the directory into you work space.
   
   2. Open predict_flask.py. Go to the part of the chart where libraries are imported. 
      Uncomment:  import tflite_runtime.interpreter as tflite  and 
      Comment:    import tensorflow.lite as tflite
      Save the changes in the file. This change is temporary. Once we build run our docker image and test our predictions and get them right, we will change it back
      
   2. Build and run the application using the commands:
      ```
      docker build -t zoomcamp-capstone-1 .
      docker run -it --rm -p 9696:9696 zoomcamp-capstone-1  (This will keep the running the prediction service from the docker container)
      ```
      
   3. Open another terminal/prompt and run predict_test.py  
      ``` 
      python predict_test.py (To test the prediction service)
      ```
      If the prediction services gives prediction, then it means the docker container is working.
      
   4. Open predict_flask.py. Go to the part of the chart where libraries are imported. 
      Comment:  import tflite_runtime.interpreter as tflite  and 
      Uncomment:    import tensorflow.lite as tflite
      Save the changes in the file.
 
