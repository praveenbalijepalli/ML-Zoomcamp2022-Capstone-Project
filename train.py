# Import Libaries
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import utils, preprocessing, models

import os
from zipfile import ZipFile

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import kaggle
 
# # Download Data

# Download data from https://www.kaggle.com/datasets/vuppalaadithyasairam/spine-fracture-prediction-from-xrays/code
# OR run the first 3 lines of code in  Cervical Fracture Modelling.ipynb

# Extracting files from the downloaded zip file
with ZipFile("spine-fracture-prediction-from-xrays.zip","r") as zip_obj:
    zip_obj.extractall()

 
train_path = "./cervical fracture/train"
test_path = "./cervical fracture/val"  # I treated this as test data and got validation data from train images



## Define Functions

### Load and Preprocess Data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### Load and Preprocess train and validation data
def load_preprocess_augment_train_val(train_path, validation_split, target_size, color_mode, class_mode, batch_size, seed ):
  
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    
    train_generator = train_datagen.flow_from_directory(train_path, # Load Data
                                                        target_size=target_size,
                                                        color_mode=color_mode,
                                                        classes=os.listdir(train_path),
                                                        class_mode=class_mode,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=seed,
                                                        subset='training')
    
    val_generator = train_datagen.flow_from_directory(train_path, # Load Data
                                                             target_size=target_size,
                                                             color_mode=color_mode,
                                                             classes=os.listdir(train_path),
                                                             class_mode=class_mode,
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             seed=seed,
                                                             subset='validation')
   
    return train_generator, val_generator


### Load and Preprocess test data
def load_preprocess_test(test_path, target_size, color_mode, class_mode, batch_size, seed):
     
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, # Load Data
                                                                            target_size=target_size,
                                                                            color_mode=color_mode,
                                                                            classes=os.listdir(test_path),
                                                                            class_mode=class_mode,
                                                                            batch_size=batch_size,
                                                                            seed=seed)
    return test_generator


## Define Model Architecture

### Define Custom Architecture

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, InputLayer, Dense, MaxPool2D, Flatten, Dropout, BatchNormalization

def custom_model(input_shape, kernel_size, pool_size, output_activation, dense_layer_1_neurons):

    custom_model = Sequential([InputLayer(input_shape=input_shape),
 
                              Conv2D(filters=64, kernel_size=kernel_size, 
                                     strides=(2, 2) , activation='relu'),
 
                              Conv2D(filters=32, kernel_size=kernel_size, 
                                     strides=(2, 2) , activation='relu'),
                              
                              Dropout(0.2),
                               
                              BatchNormalization(), 
                               
                              MaxPool2D(pool_size=pool_size),
                               
                              Flatten(),
 
                              Dense(units=dense_layer_1_neurons , activation='relu'),
                                                                                        
                              Dense(units=1, activation=output_activation)], name="Custom_model")
    
    print(custom_model.summary())
    
    return custom_model


## Compile and Train Model with callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def train_model(model, train_generator, val_generator, epochs):
    
    filepath=f'cerv_fracture_model.h5'     
    rlrp = ReduceLROnPlateau(monitor="val_accuracy", factor=0.1, patience=2)
    checkpoint = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode='max')

    callback = [checkpoint,rlrp]
         
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        
    history = model.fit_generator(train_generator, epochs=epochs, 
                                  validation_data=val_generator, shuffle=True,
                                  callbacks=callback)
        
    return model, history
 
 
# Evaluate Model
def eval_model(model, test_generator):
    eval_metrics = model.evaluate(test_generator, return_dict=True) 
    return eval_metrics

 
# Model Pipeline
def model_pipeline(model):
    
    # Load and Preprocess Data
    train_generator, val_generator = load_preprocess_augment_train_val(train_path, validation_split, target_size, 
                                                                         color_mode, class_mode, batch_size, seed)
    print(f"Train Classes index:  {train_generator.class_indices}\n")
    
    test_generator = load_preprocess_test(test_path, target_size, color_mode, class_mode, batch_size, seed)
    
    print(f"Test Classes index:  {test_generator.class_indices}\n")
     
    # Train Models
    model, history = train_model(model, train_generator, val_generator, epochs) 
    
    print()
    
    # Evaluate Models
    eval_metrics = eval_model(model, test_generator)
  
    return model, history, eval_metrics 
 
 
 
# Load and Preprocess Parameters
train_path = "./cervical fracture/train"
test_path = "./cervical fracture/val" 
target_size=(224, 224)
color_mode="rgb"
class_mode='binary'
batch_size=16
seed=100
validation_split=0.20

# Custom Model Parameters
input_shape = (224, 224, 3)
kernel_size = (3, 3)
pool_size = (2, 2)
output_activation = 'sigmoid'
dense_layer_1_neurons = 64

# Model Compilation Parameters
learning_rate = 0.001
metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
loss_fn = 'binary_crossentropy'
optimizer = 'SGD' 

## Parameters to Train and Evaluate Model
epochs = 20

 
# Define custom model 
custom_model = custom_model(input_shape, kernel_size, pool_size,output_activation, dense_layer_1_neurons)

# Run the model pipeline to train
model, history, eval_metrics =  model_pipeline(custom_model)

# Load Best Model
best_model = tf.keras.models.load_model('cerv_fracture_model.h5')

# Save Model
# tf.keras.models.save_model(best_model,'cerv_fracture_model.h5')

