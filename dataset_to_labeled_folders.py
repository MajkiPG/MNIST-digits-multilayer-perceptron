import numpy as np
import skimage.io as im
import os
import shutil
import sys

from functions.progress_bar import print_progress

#DIRECTORIES AND FILES SETUP
dataset_raw_location = "./dataset/"
datset_images_location = "./dataset_images"
images_file_name = "train-images-idx3-ubyte"
labels_file_name = "train-labels-idx1-ubyte"

#MAGIC NUMBERS VALS
check_magic_number_images = 2051
check_magic_number_labels = 2049

#Create grouped directories for dataset images.
if os.path.isdir(datset_images_location):
    shutil.rmtree(datset_images_location)
os.mkdir(datset_images_location)
for x in range(10):
    os.mkdir(datset_images_location + "/" + str(x))

#Open and read dataset files.
file_images = open(dataset_raw_location+images_file_name, "rb")
file_labels = open(dataset_raw_location+labels_file_name, "rb")
raw_images_data = file_images.read()
raw_labels_data = file_labels.read()

#Initial values for dataset variables.
magic_number_images = 0
magic_number_labels = 0
number_of_items_images = 0
number_of_items_labels = 0
number_of_rows = 0
number_of_columns = 0
pointer_data_images = 0
pointer_data_labels = 0

#Read and check magic numbers (32-bit integers).
for x in range(4):
    magic_number_images = (magic_number_images << 8) + raw_images_data[x]
    magic_number_labels = (magic_number_labels << 8) + raw_labels_data[x]
if check_magic_number_images != magic_number_images or check_magic_number_labels != magic_number_labels:
    sys.exit("Incorrect magic numbers. Dataset might be corrupted.")
pointer_data_images += 4
pointer_data_labels += 4

#Read and check items count (32-bit integers).
for x in range(4, 8):
    number_of_items_images = (number_of_items_images << 8) + raw_images_data[x]
    number_of_items_labels = (number_of_items_labels << 8) + raw_labels_data[x]
if number_of_items_images != number_of_items_labels:
    sys.exit("Different items count. Dataset might be corrupted.")
pointer_data_images += 4
pointer_data_labels += 4

#Read number of columns in image (32-bit integer).
for x in range(8, 12):
    number_of_rows = (number_of_rows << 8) + raw_images_data[x]
pointer_data_images += 4

#Read number of rows in image (32-bit integer).
for x in range(12, 16):
    number_of_columns = (number_of_columns << 8) + raw_images_data[x]
pointer_data_images += 4

print_progress(0, number_of_items_images, prefix="Labeling images: ", suffix="complete", bar_length=100)

#Read and save images to adequate folders. Unsigned byte per pixel. Number_of_columns * number_of_rows pixels per image. Pixels organized row-wise.
for image_number in range(number_of_items_images):
    image = 255 * np.ones(shape=[number_of_rows, number_of_columns, 1], dtype=np.uint8)
    for column in range(number_of_columns):
        for row in range(number_of_rows):
            image[column][row] = ~raw_images_data[pointer_data_images]
            pointer_data_images += 1
    #cv.imwrite(datset_images_location + "/" + str(raw_labels_data[pointer_data_labels]) + "/" + str(image_number)+".png", image)
    im.imsave(datset_images_location + "/" + str(raw_labels_data[pointer_data_labels]) + "/" + str(image_number)+".png", image)
    pointer_data_labels += 1
    print_progress(image_number, number_of_items_images, prefix="Labeling images: ", suffix="complete", bar_length=100)