import pandas as pd
import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Flatten, Dense, Dropout)
import matplotlib.pyplot as plt

class Callback(tf.keras.callbacks.Callback):
    epoch_controller = 25

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if (self.epoch%self.epoch_controller==0):
            print('Epoch: ' + str(self.epoch) + ' loss: ' + str(logs['loss']))
        
def dnn_model(X_train, 
          Y_train, 
          neurons_per_layer=1000,
          activation_per_layer='relu',
          epochs=500, 
          dropout_rate=0.25, 
          loss='huber', 
          l1_regularizer=False,
          batch_size=32,
          verbose=0):
    """
    create and return the desired model
    """
    #regularizer
    if l1_regularizer:
        regularizer=tf.keras.regularizers.L1(l1=0.01)
    else:
        regularizer=None
    
    model = keras.models.Sequential([
        Flatten(input_shape=X.shape[1:]),
        Dense(neurons_per_layer, activation=activation_per_layer, kernel_regularizer=regularizer),
        Dense(neurons_per_layer, activation=activation_per_layer, kernel_regularizer=regularizer),
        Dense(neurons_per_layer, activation=activation_per_layer, kernel_regularizer=regularizer),
        Dropout(dropout_rate),
        Dense(1)
    ])
            
    #define loss to minimize
    if loss=='huber':
        loss_to_minimize = tf.keras.losses.Huber() 
    elif loss=='mse':
        loss_to_minimize = tf.keras.losses.MeanSquaredError()
    
    #optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, 
        amsgrad=False, name='Adam'
    )
    
    #compile the model
    model.compile(loss=loss_to_minimize, 
                  optimizer=optimizer)
    
    #model fit
    model_history = model.fit(X_train, Y_train, 
                              batch_size=batch_size, 
                              epochs=epochs,  
                              callbacks=[Callback()], 
                              verbose=verbose)
    
    return model_history, model


def train_predict(X, Y, **model_params):
    """
    Train and predict pixel values from the model
    """
    neurons_per_layer = model_params["neurons_per_layer"] if "neurons_per_layer" in model_params else 100
    activation_per_layer = model_params["activation_per_layer"] if "activation_per_layer" in model_params else "relu"
    epochs = model_params["epochs"] if "epochs" in model_params else 100
    dropout_rate = model_params["dropout_rate"] if "dropout_rate" in model_params else 0.25
    loss = model_params["loss"] if "loss" in model_params else "mse"
    l1_regularizer = model_params["l1_regularizer"] if "l1_regularizer" in model_params else True
    verbose = model_params["verbose"] if "verbose" in model_params else 0
    
    model_history, model = dnn_model(X, Y,
                                     neurons_per_layer=neurons_per_layer,
                                     activation_per_layer=activation_per_layer, 
                                     epochs=epochs, 
                                     dropout_rate=dropout_rate, 
                                     loss=loss, 
                                     l1_regularizer=l1_regularizer, 
                                     verbose=verbose)
    y_pred = model.predict(X)
    
    plt.imshow(np.reshape(y_pred, (95, 79)), cmap='gray') #display the recovered image
    
    return model, model_history, y_pred
