import numpy as np
from skimage.io import imread
import keras_preprocessing as prep
import os

def load_image_data(dataset_dir, flatten=False):
    """
    Loads and shuffles image dataset from label grouped folders.
    @params:
        dataset_dir - Required  : path to dataset (Str)
        flatten     - Optional  : flatten image into 1-D array for fully connected network (Bool)
    """
    images = []
    labels = []
    for folder in range(10):
        folder_path = dataset_dir + str(folder)
        for image in os.listdir(folder_path):
            current_image =  imread(folder_path + "/" + image)
            current_image_arr = prep.image.img_to_array(current_image)
            if flatten:
                current_image_arr = current_image_arr.flatten()
            images.append(current_image_arr)
            labels.append(folder)
    pointers = np.arange(len(labels))
    np.random.shuffle(pointers)
    return np.array(images)[pointers], np.array(labels)[pointers]