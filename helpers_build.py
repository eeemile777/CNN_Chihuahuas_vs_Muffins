from tensorflow.keras import models, layers
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import os 
from tensorflow import keras
import matplotlib.pyplot as plt
import visualkeras
from tensorflow.keras.utils import plot_model
import logging

import re
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam


from tensorflow.keras import models, layers
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Precision, Recall, AUC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization



def build_ffnn_model(neurons_list, 
                     input_shape,
                     activation='relu',
                     output_activation='sigmoid',
                     optimizer='adam',
                     loss='binary_crossentropy', 
                     metrics=['accuracy'],
                     dropout_rates=None,
                     batch_norms=None, 
                     learning_rate=0.001):
    """
    Build a Feedforward Neural Network (FFNN) model.

    Parameters:
    - neurons_list (list): List of integers representing the number of neurons in each hidden layer.
    - input_shape (tuple): Shape of the input data (excluding batch size).
    - activation (str): Activation function for hidden layers.
    - output_activation (str): Activation function for the output layer.
    - optimizer (str): Optimization algorithm.
    - loss (str): Loss function.
    - metrics (list): List of metrics to monitor during training.
    - dropout_rates (list): List of dropout rates for regularization (None for no dropout).
    - batch_norms (list): List of booleans indicating whether to include batch normalization for each layer.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - model (Sequential): Compiled FFNN model.
    """

    # to make a unique name based on architecture 
    dense_layers_info = f'_dense_layers_{"_".join(map(str, neurons_list))}'
    model_name = f'ffnn_model_{dense_layers_info}'

    model = models.Sequential(name=model_name)

    # we have to flatten the input for FFNN
    model.add(layers.Flatten(input_shape=input_shape, name='Flatten_layer'))

    # to add dense layers
    for i, (neurons, dropout_rate, batch_norm) in enumerate(zip(neurons_list, dropout_rates or [], batch_norms or []), start=1):
        model.add(layers.Dense(neurons, activation=activation, name=f'hidden_dense_layer_{i}'))

        # to add Dropout layer for regularization if dropout rate is provided
        if dropout_rate is not None:
            model.add(layers.Dropout(rate=dropout_rate, name=f'dropout_{i}'))

        # to add Batch Normalization layer if specified
        if batch_norm:
            model.add(layers.BatchNormalization(name=f'batch_norm_{i}'))

    # output layer with specified activation for binary classification
    model.add(layers.Dense(1, activation=output_activation, name='output_layer'))

    # Compile the model with specified optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # to set the learning rate i had to use Keras backend
    model.optimizer.learning_rate = learning_rate

    model.summary()

    return model





def build_cnn_model(neurons_list,
                    filters_list, 
                    input_shape,
                    activation='relu',
                    output_activation='sigmoid',
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                    conv_dropout_rates=None, 
                    conv_batch_norms=None,
                    dense_dropout_rates=None,
                    dense_batch_norms=None,
                    learning_rate=0.001):
    """
    Build a Convolutional Neural Network (CNN) model.

    Parameters:
    - neurons_list : List of integers representing the number of neurons in each dense layer.
    - filters_list : List of integers representing the number of filters in each convolutional layer.
    - input_shape : Shape of the input data (excluding batch size).
    - activation (str): Activation function for convolutional and dense layers.
    - output_activation (str): Activation function for the output layer.
    - optimizer (str): Optimization algorithm.
    - loss (str): Loss function.
    - metrics (list): List of metrics to monitor during training.
    - conv_dropout_rates (list): List of dropout rates for convolutional layers (None for no dropout).
    - conv_batch_norms (list): List of booleans indicating whether to include batch normalization for each convolutional layer.
    - dense_dropout_rates (list): List of dropout rates for dense layers (None for no dropout).
    - dense_batch_norms (list): List of booleans indicating whether to include batch normalization for each dense layer.
    - learning_rate (float): Learning rate for the optimizer.

    Returns:
    - model (Sequential): Compiled CNN model.
    """
    
    # to make a unique name based on architecture parameters
    conv_layers_info = f'_conv_layers_{"_".join(map(str, filters_list))}'
    dense_layers_info = f'_dense_layers_{"_".join(map(str, neurons_list))}'
    
    # Add dropout rates and batch normalization flags to the model name
    if conv_dropout_rates and any(val is not None for val in conv_dropout_rates):
        conv_layers_info += '_dropout'
    if conv_batch_norms and any(val for val in conv_batch_norms):
        conv_layers_info += '_batch_norm'
    if dense_dropout_rates and any(val is not None for val in dense_dropout_rates):
        dense_layers_info += '_dropout'
    if dense_batch_norms and any(val for val in dense_batch_norms):
        dense_layers_info += '_batch_norm'
    
    
    model_name = f'cnn_model_{conv_layers_info}_{dense_layers_info}'
    
    model = models.Sequential(name=model_name)

    # Convolutional layers
    for i, filters in enumerate(filters_list):
        model.add(Conv2D(filters, (3, 3), activation=activation, input_shape=input_shape))
        if conv_batch_norms and conv_batch_norms[i]:
            model.add(BatchNormalization())
        if conv_dropout_rates and conv_dropout_rates[i]:
            model.add(Dropout(conv_dropout_rates[i]))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Dense layers
    for i, neurons in enumerate(neurons_list):
        model.add(Dense(neurons, activation=activation))
        if dense_batch_norms and dense_batch_norms[i]:
            model.add(BatchNormalization())
        if dense_dropout_rates and dense_dropout_rates[i]:
            model.add(Dropout(dense_dropout_rates[i]))

    # Output layer
    model.add(Dense(1, activation=output_activation))

    # Configure the optimizer with the specified learning rate
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate)
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
    elif optimizer.lower() == 'adagrad':
        opt = tf.keras.optimizers.legacy.Adagrad(learning_rate=learning_rate)
    elif optimizer.lower() == 'adadelta':
        opt = tf.keras.optimizers.legacy.Adadelta(learning_rate=learning_rate)
    elif optimizer.lower() == 'adamax':
        opt = tf.keras.optimizers.legacy.Adamax(learning_rate=learning_rate)
    elif optimizer.lower() == 'nadam':
        opt = tf.keras.optimizers.legacy.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer. Please choose from 'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'.")


    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    #saving the summary as .txt file
    
    with open(os.path.join('untrained_models/summaries', f"summary_{model_name}.txt"), 'w') as summary_file:
        model.summary(print_fn=lambda x: summary_file.write(x + '\n'))

    return model





def get_model_size(model_path):
    # get the size of the model file in bytes
    return os.path.getsize(model_path)



def build_models_from_dict(architectures_dict, input_shape_cnn, save_folder='untrained_models'):
    """
    Build and save untrained CNN models based on configurations provided in a dictionary.

    Parameters:
    - architectures_dict (dict): Dictionary containing configurations for building CNN models.
    - input_shape_cnn (tuple): Input shape for the CNN models.
    - save_folder (str): Folder to save untrained models. Default is 'untrained_models'.

    Returns:
    dict: Dictionary containing built CNN models.

    Model Configuration Parameters:
    - neurons_list_cnn (list): List of neurons for each dense layer in the CNN model.
    - filters_list_cnn (list): List of filters for each convolutional layer in the CNN model.
    - conv_dropout_rates_cnn (list): List of dropout rates for convolutional layers.
    - conv_batch_norms_cnn (list): List of batch normalization settings for convolutional layers.
    - dense_dropout_rates_cnn (list): List of dropout rates for dense layers.
    - dense_batch_norms_cnn (list): List of batch normalization settings for dense layers.
    - Other parameters: Fixed settings for activation, optimizer, loss, metrics, and learning rate.

    Note:
    - The untrained models are saved in the specified folder.
    - The function returns a dictionary with model names as keys and corresponding model objects.
    - The models are sorted based on their sizes before returning the dictionary.
    """
    models_dict = {}


    os.makedirs(save_folder, exist_ok=True)

    model_sizes = {}  # dictg to store model names and their corresponding sizes

    for arch_name, arch_params in architectures_dict.items():
        try:
            model = build_cnn_model(
                arch_params['neurons_list_cnn'],
                arch_params['filters_list_cnn'],
                input_shape_cnn,
                activation='relu',
                output_activation='sigmoid',
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
                conv_dropout_rates=arch_params['conv_dropout_rates_cnn'],
                conv_batch_norms=arch_params['conv_batch_norms_cnn'],
                dense_dropout_rates=arch_params['dense_dropout_rates_cnn'],
                dense_batch_norms=arch_params['dense_batch_norms_cnn'],
                learning_rate=0.001
            )

            # to save the untrained model to the specified folder in  Keras 
            model_path = os.path.join(save_folder, f'untrained_{model.name}.keras')
            model.save(model_path)

            model_size = get_model_size(model_path)

            
            models_dict[model.name] = model

            print(f"{arch_name}: Model size - {model_size} bytes")
        except Exception as e:
            print(f"Error building model: {str(e)}")
            continue

    # sort models_dict based on model sizes
    models_dict = dict(sorted(models_dict.items(), key=lambda item: get_model_size(os.path.join(save_folder, f'untrained_{item[0]}.keras'))))
    # no need for them to stay in the folder (less memory no?)
    for model_name in models_dict:
        os.remove(os.path.join(save_folder, f'untrained_{model_name}.keras'))

    return models_dict








def generate_architectures(batchnorm=False, drop=None):
    """
    Generate a dictionary of convolutional and dense architectures with varying layer configurations.

    Parameters:
    - batchnorm (bool): If True, includes batch normalization in the architectures.
    - drop (float): Dropout rate to be applied in the architectures.

    Returns:
    dict: A dictionary containing merged convolutional and dense architectures.
    
    Architecture Structure:
    - Convolutional Architectures: Configurations for convolutional layers with varying filters, dropout rates, and batch normalization.
    - Dense Architectures: Configurations for dense layers with varying neurons, dropout rates, and batch normalization.
    - Merged Architectures: Combined configurations of convolutional and dense layers for each architecture.

    Note:
    The function generates architectures with four layers, each with an increasing number of filters/neurons.
    The architectures are created for specified values [16, 32, 64, 128, 256] and merged into a final dictionary.
    """
    values = [16, 32, 64, 128, 256]
    dense_architectures = {}
    conv_architectures = {}
    num_layers = 4

    for value in values:
        conv_architectures_for_value = {}
        dense_architectures_for_value = {}
        for i in range(num_layers):
            architecture_key = f'_{i + 1}'
            filters_list_cnn = [value * 2**j for j in range(i + 1)]
            neurons_list_cnn = [value * 2**j for j in range(i + 1)]

            conv_architectures_for_value[architecture_key] = {
                'filters_list_cnn': filters_list_cnn,
                'conv_dropout_rates_cnn': [drop] * (i + 1),
                'conv_batch_norms_cnn': [batchnorm] * (i + 1)
            }
            dense_architectures_for_value[architecture_key] = {
                'neurons_list_cnn': neurons_list_cnn,
                'dense_dropout_rates_cnn': [drop] * (i + 1),
                'dense_batch_norms_cnn': [batchnorm] * (i + 1)
            }
    
        conv_architectures[f'_{value}'] = conv_architectures_for_value
        dense_architectures[f'_{value}'] = dense_architectures_for_value
        
    
    merged_architectures = {}
    arch = 0

    for conv_key, conv_value in conv_architectures.items():
        for neurons_key, neurons_value in dense_architectures.items():
            for c, b in neurons_value.items():
                for v, j in conv_value.items():
                    arch+=1
                    merged_key = f"{arch}"
                    merged_architectures[merged_key] = {
                        **j,
                        **b
                    }

    return merged_architectures


def config_extract(input_string):
    """
    Extract filter and neuron configurations from a given input string.

    Parameters:
    - input_string (str): A string containing configuration information for convolutional and dense layers.

    Returns:
    tuple: A tuple containing two lists - the first for convolutional layer filters and the second for dense layer neurons.

    Configuration Structure:
    The input string is expected to follow the pattern 'conv_layers_X_Y_dense_layers_Z_W',
    where X, Y, Z, W represent the numbers of convolutional and dense layers, respectively.

    Example:
    If input_string is 'conv_layers_64_128_256_dense_layers_512_256_128',
    the function will return ([64, 128, 256], [512, 256, 128]).
    """
    filters = []
    neurones = []

    #  the pattern to match the numbers
    pattern = r'conv_layers_([\d_]+).*dense_layers_([\d_]+)'

    #  to find matches
    match = re.search(pattern, input_string)

    if match:
        filters = [int(num) for num in match.group(1).split('_') if num]
        neurones = [int(num) for num in match.group(2).split('_') if num]

    return (filters, neurones)


