Problem Statement:

Cervical Spinal Fracture Classification from C.T. scans using Transfer Learning / Custom Deep Convolutional Neural Network

 The fractures to cervical region usually result from high energy trauma like automobile crashes and falls. In elderly people, a fall of the charir or falling on the ground can cause a cervical fracture. Cervical fracture is the fracture of any of the seven cervical vertebrae in the neck. Since they support the head and connect it to the shoulders and the body, immediate response to an injury to it paramount as it can have serious consequences. Injury to the vertebra can lead temporary or permanent paralysis from the neck down and in some cases even leading to death. So, a physician will usually need support from radiographic studies such as MRI or CT scans to determine the extent of the injuries. This project is an endeavour to use AI to assist a physician to determine if a CT scan image shows a "fracture" in a vertbrae or if it is "normal".

``` 
Artificial Intelligence Domain
Domain             : Computer Vision
Sub-Domain         : Deep Learning, Image Classification
Architectures      : Deep CNNs, Inceptionv3, Resnet50V2, Xception
Application        : Medical Image Classification
```


Dataset Details
```
Dataset Description: Contains images of Fractured and Normal Cervical CT scans in their respective folders in Train and Test Folders
Dataset Name       : Spine Fracture Prediction from C.T. Dataset
Dataset Link       : [Spine fracture prediction from C.T Dataset)(Kaggle)](https://www.kaggle.com/datasets/vuppalaadithyasairam/spine-fracture-prediction-from-xrays/code)
Dataset Size       : 311 MB 
Number of Classes  : 2 (Fracture, Normal)
Number of Images   : Training Images - 3800 (Training  - 3040 and Validation - 760 Images
                     Testing Images - 400
```

Parameters for Training
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

Classification Metrics of the Final Model
```
Final Model             :
Train, Val and Test     
Accuracy (F-1) Score    : Train  -     , Val -    , Test - 
Loss                    : Train  -     , Val -    , Test -   
Precision               : Train  -     , Val -    , Test -
Recall (Pneumonia)      : Train  -     , Val -    , Test -
```

Sample Output:
```

```
Confusion Matrix:
```

```

Tools / Libraries
```
Languages               : Python
Tools/IDE               : Anaconda
Libraries               : Keras, TensorFlow
Virtual Environment     : pipenv
```
```
Train Script            :
Keras to TFlite Script  :
Predict Script          :
Test Script             :
```

Run the Model as is: 
Steps to run the scripts/notebooks as is:

    1. Clone the repo by running the following command:

       `git clone ` 

    2. Open a terminal or command prompt and change directory to the folder where this project is cloned.

    3. Run the following command to activate the virtual environment for the project:

       `pipenv shell`

       In case, pipenv is not installed in your system, to install pipenv and to activate the virtual environment for the project, type the following commands:

       ```
       pip install pipenv` 
       pipenv shell` (in the project folder)
       ``` 
    4.  To install the files and dependencies related to the project, run the following in the folder containing Pipfile/Pipfile.lock

       `pipenv install`

    5.  To run the scripts do the following:
        
        a. To train the the model and save it using train.py script, run the following command in the terminal/prompt.

        `python train.py` (To train and save the model)
        
        b. Run predict-flask.py using python in a terminal/prompt.

        `python predict-flask.py` (To start the prediction service)

        c. Open another terminal/prompt and run predict-test.py Or run predict-test.ipynb jupyter notebook to test the prediction service.

        python predict-test.py (To test the prediction service)



Model as Web Service:

Using Waitress
How to run the application:
1. Docker

Step 1: Clone the GitHub repository with the project.

git clone https://github.com/jvscursulim/mlzoomcamp_capstone_project

Step 2: Access the GitHub repository folder.

cd mlzoomcamp_capstone_project

Step 3: Create the docker image.

docker build -t surface_crack_detection .

Step 4: Run the application with docker.

docker run -p 4242:4242 surface_crack_detection

Step 5: Put some images that you want to classify in imgs folder

Step 6: Change the value associated with the key image_path of the dictionary assigned with the variable img inside test.py file. The new value must be the path of one of the images inside imgs folder. (Please refer to section "How to send data for the application")
2. Without Docker

Step 1: Clone the GitHub repository with the project.

git clone https://github.com/jvscursulim/mlzoomcamp_capstone_project

Step 2: Access the GitHub repository folder.

cd mlzoomcamp_capstone_project

Step 3: Create a virtual environment.

python -m venv env

Step 4: Activate your virtual environment.

    Linux: Activation of the virtual environment.

source env/bin/activate

    Windows: Activation of the virtual environment.

env/Scripts/Activate.ps1

Step 5: Install pipenv.

pip install pipenv

Step 6: Install the packages required for this application using the command below.

pipenv install

Step 7: Run the application with gunicorn.

gunicorn --bind=0.0.0.0:4242 predict:app

Step 8: Put some images that you want to classify in imgs folder

Step 9: Change the value associated with the key image_path of the dictionary assigned with the variable img inside test.py file. The new value must be the path of one of the images inside imgs folder. (Please refer to section "How to send data for the application")
Observation: If you want to train a model

    https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

    Observation: Download the dataset using the link above and extract the files in data/dataset before start the training.

Access script folder.

cd script

Make your changes in train.py and run the file using the command below.

python train.py

After training process, you can build the new application following the instructions in the sections 1. Docker and 2. Without Docker.
How to send data for the application:

For instance the test.py file.
Code snippet

import requests

img = {"image_path": "/workspaces/mlzoomcamp_capstone_project/imgs/00001.jpg"}

url = "http://localhost:4242/predict"
print(requests.post(url, json=img).json())