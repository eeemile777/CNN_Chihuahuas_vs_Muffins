import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import os 
from tensorflow import keras
import matplotlib.pyplot as plt
import visualkeras
from tensorflow.keras.utils import plot_model
import logging
from skimage import transform, util

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import cv2


def process_image(img_path, class_num, img_size, data):
    """
    Process an image by loading, resizing, and appending it to the appropriate dataset.

    Parameters:
    - img_path (str): Path to the image file.
    - class_num (int): Numeric label for the image class.
    - img_size (int): Size for resizing the image.
    - data (list): The list to append the processed image and label.

    Returns:
    None
    """
    try:
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        #  if the image is successfully loaded
        if img_array is None:
            print(f"Error loading image: {img_path}")
            return

        # resize the image
        new_array = cv2.resize(img_array, (img_size, img_size))

        # add to the dataset
        data.append([new_array, class_num])

    except Exception as e:
        print(f"Error processing image: {img_path}, Error: {str(e)}")

def get_data(directory, categories, img_size):
    """
    Collect and process images from a specified directory for each category.

    Parameters:
    - directory (str): The main directory containing subdirectories for each category.
    - categories (list): List of category names.
    - img_size (int): Size for resizing the image.

    Returns:
    list: The collected and processed data.
    """
    data = []
    for category in categories:
        path = os.path.join(directory, category)
        class_num = categories.index(category)

        for img in os.listdir(path):
            if img.endswith(".jpg"):
                img_path = os.path.join(path, img)
                process_image(img_path, class_num, img_size, data)
    
    return data


def apply_augmentation(image, label):
    """
    Apply various image augmentation techniques to enhance the diversity of a dataset.

    Parameters:
    - image (numpy.ndarray): The input image to be augmented.
    - label: The label associated with the input image.

    Returns:
    list: A list of tuples containing original and augmented images and their corresponding labels.
    
    Augmentation Techniques:
    1. Original Image: The input image with no transformation.
    2. Rotated Image: The input image rotated by a random angle (-30 to 30 degrees).
    3. Horizontally Flipped Image: The input image flipped horizontally.
    4. Vertically Flipped Image: The input image flipped vertically.
    5. Noisy Image: The input image with added Gaussian noise.
    6. Zoomed Image: The input image with a random zoom factor applied.
    """
    augmented_images = []

    # Reshape the image to (64, 64) not 64,64,1
    reshaped_image = image.squeeze()

    # Add the original image and its label as the first element
    augmented_images.append((reshaped_image, label))

    # Rotate the image by a random angle (-30 to 30 degrees)
    rotated_image = transform.rotate(reshaped_image, angle=np.random.uniform(-30, 30))
    rotated_image = transform.resize(rotated_image, (64, 64))  # Resize to (64, 64)
    augmented_images.append((rotated_image, label))

    # Flip the image horizontally 
    flipped = np.fliplr(reshaped_image)
    flipped = transform.resize(flipped, (64, 64))  # Resize to (64, 64)
    augmented_images.append((flipped, label))

    # Flip the image vertically 

    flipped = np.flipud(reshaped_image)
    flipped = transform.resize(flipped, (64, 64))  # Resize to (64, 64)
    augmented_images.append((flipped, label))

    # Add Gaussian noise to the image
    noisy_image = util.random_noise(reshaped_image, mode='gaussian', seed=None, clip=True)
    augmented_images.append((noisy_image, label))

    # Apply random zoom
    zoom_factor = np.random.uniform(0.8, 1.2)
    zoomed_image = transform.rescale(reshaped_image, zoom_factor, mode='constant')
    zoomed_image = transform.resize(zoomed_image, (64, 64))  # Resize to (64, 64)
    augmented_images.append((zoomed_image, label))

    return augmented_images



