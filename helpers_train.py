import datetime
import matplotlib.pylab as plt
import numpy as np
import os 
from tensorflow import keras
import visualkeras
from tensorflow.keras.utils import plot_model
import logging


import tensorflow as tf
import seaborn as sns

from tensorflow.keras import callbacks
import datetime
import time


def train_model(model, x, y, batch_size=32, epochs=15, validation_split=0.2):
    """
    Train a given model using the specified data.

    Parameters:
    - model: The Keras model to be trained.
    - x: Input data for training.
    - y: Target labels for training.
    - batch_size (int): Batch size for training. Default is 32.
    - epochs (int): Number of epochs for training. Default is 15.
    - validation_split (float): Fraction of training data to be used as validation data. Default is 0.2.

    Returns:
    dict: Training history containing metrics for each epoch.

    Training Callback:
    - EarlyStopping: Monitors validation loss and stops training if no improvement is observed.

    Note:
    - The function prints the training time and the device used for training.
    """
    # extract the model name
    model_name = model.name if model.name else 'unnamed_model'

    # set up EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)


    start_time = time.time()
    # train the model with validation split
    history = model.fit(
        x,
        y,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=batch_size,
        callbacks= early_stopping
    ).history

    end_time = time.time()
    print(f'Training Time (GPU): {end_time - start_time} seconds')
    print("Device used for training:", model._train_counter.device)

    return history




# to train more than 1 model


def train_models(models_dict, x_train, y_train, batch_size=32, epochs=15, validation_split=0.2):
    """
    Train multiple models using the specified data.

    Parameters:
    - models_dict (dict): Dictionary containing Keras models to be trained.
    - x_train: Input data for training.
    - y_train: Target labels for training.
    - batch_size (int): Batch size for training. Default is 32.
    - epochs (int): Number of epochs for training. Default is 15.
    - validation_split (float): Fraction of training data to be used as validation data. Default is 0.2.

    Returns:
    dict: Dictionary containing training histories for each model.

    Training Callback:
    - EarlyStopping: Monitors validation loss and stops training if no improvement is observed.

    Note:
    - The function prints progress information during training.
    - Training histories are saved to pickle files in the "histories" folder.
    """
    histories_folder = 'histories'
    os.makedirs(histories_folder, exist_ok=True)

    histories_dict = {}

    total_models = len(models_dict)
    completed_models = 0

    for model_name, model in models_dict.items():
        completed_models += 1

        # alculate progress percentage
        progress_percentage = (completed_models / total_models) * 100

        print(f"Training Model {completed_models}/{total_models} ({progress_percentage:.2f}%): {model_name}")

        history = train_model(
            model,
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
        )

        histories_dict[model_name] = history

        # save the history to a pickle file in the "histories" folder
        history_save_path = os.path.join(histories_folder, f'history_{model_name}.pkl')
        with open(history_save_path, 'wb') as f:
            pickle.dump(history, f)

    return histories_dict








def plot_training_history(history, title=''):
    """
    Plot training history metrics (accuracy and loss) for a given model.

    Parameters:
    - history (dict): Training history containing metrics for each epoch.
    - title (str): Title to be displayed on the plots.

    Plots:
    - Accuracy Over Epochs
    - Loss Over Epochs
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot accuracy
    ax[0].plot(history['accuracy'], label='train acc')
    ax[0].plot(history['val_accuracy'], label='val acc')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epochs')
    ax[0].legend()
    ax[0].set_title(f'Accuracy Over Epochs\n{title}') 

    # Plot loss
    ax[1].plot(history['loss'], label='train loss')
    ax[1].plot(history['val_loss'], label='val loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epochs')
    ax[1].legend()
    ax[1].set_title(f'Loss Over Epochs\n{title}')  
    plt.show()

