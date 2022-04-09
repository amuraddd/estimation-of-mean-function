import pandas as pd
import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Flatten, BatchNormalization, Activation, Dense, Dropout)
import matplotlib.pyplot as plt

class Callback(tf.keras.callbacks.Callback):
    epoch_controller = 25

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if (self.epoch%self.epoch_controller==0):
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
        
def simple_dnn_model(X_train, 
                        Y_train, 
                        neurons_per_layer=1000,
                        activation_per_layer='relu',
                        epochs=500,
                        batch_size=8,
                        verbose=0):
    """
    DNN with default RELU activation
    """        
    
    model = keras.models.Sequential([
        Flatten(input_shape=X_train.shape[1:]),
        Dense(neurons_per_layer),
        Activation(activation_per_layer),
        Dense(neurons_per_layer),
        Activation(activation_per_layer),
        # Dense(neurons_per_layer),
        # Activation(activation_per_layer),
        Dense(1)
    ])
            
    #define loss to minimize
    loss_to_minimize = tf.keras.losses.MeanSquaredError()
    
    #compile the model
    model.compile(loss=loss_to_minimize)
    
    #model fit
    model_history = model.fit(X_train, Y_train, 
                              batch_size=batch_size, 
                              epochs=epochs,  
                              callbacks=[Callback()], 
                              verbose=verbose)
    
    return model_history, model


def train_predict_simple_dnn_model(X, Y, **model_params):
    """
    Train and predict pixel values from the model
    """
    neurons_per_layer = model_params["neurons_per_layer"] if "neurons_per_layer" in model_params else 100
    activation_per_layer = model_params["activation_per_layer"] if "activation_per_layer" in model_params else "relu"
    epochs = model_params["epochs"] if "epochs" in model_params else 500
    batch_size = model_params["batch_size"] if "batch_size" in model_params else 8
    verbose = model_params["verbose"] if "verbose" in model_params else 0
    
    model_history, model = simple_dnn_model(X, Y,
                                     neurons_per_layer=neurons_per_layer,
                                     activation_per_layer=activation_per_layer, 
                                     epochs=epochs, 
                                     batch_size=batch_size,
                                     verbose=verbose)
    y_pred = model.predict(X)
    
    plt.imshow(np.reshape(y_pred, (95, 79)), cmap='gray') #display the recovered image
    
    return model_history, model, y_pred
