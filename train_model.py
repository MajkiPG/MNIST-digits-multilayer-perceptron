import os
import numpy as np
import time

from skimage.io import imread
from sklearn.model_selection import train_test_split
from functions.dataset import load_image_data
from functions.neural_model import compute_outputs, classify
from functions.backpropagation import backpropagate
from functions.progress_bar import print_progress

#DIRECTORIES SETUP
dataset_dir = "./dataset_images/"

#LOAD DATA, TRAIN/TEST SPLIT, GENERATE ONE-HOT LABELS FROM TRAINING LABELS
print("Loading dataset to memory \n")
images, labels = load_image_data(dataset_dir, flatten=True)
images_train, images_test, labels_train, label_test = train_test_split(images, labels)
one_hot_labels = np.eye(10)[labels_train]

#DATA NORMALIZATION TO 0-1 RANGE
images_train = images_train/255
images_test = images_test/255

#MODEL DEFINITION
input_size = 28*28
output_size = 10
layers = [256, 128]
learning_rate = 0.1
epochs = 3

#INITIALIZE WEIGHTS MATRICES WITH VALUES FROM -1 TO 1 RANGE
layers.append(output_size)
weights_matrices = []
for i, layer in enumerate(layers, start=0):
    if i == 0:
        weights_matrices.append(np.random.uniform(low=-1, high=1, size=(layer, input_size)))
    else:
        weights_matrices.append(np.random.uniform(low=-1, high=1, size=(layer, layers[i-1])))

#TRAIN MODEL
print("Training model: \n")
for epoch in range(epochs):
    start_time = time.time()
    for i in range(len(images_train)):
        outputs = compute_outputs(weights_matrices, images_train[i])
        weights_matrices = backpropagate(weights_matrices, outputs, images_train[i], one_hot_labels[i], learning_rate)
        print_progress(i, len(images_train), prefix=("Epoch: " + str(epoch+1))+"/" + str(epochs), suffix="complete", bar_length=50)
    score = 0
    end_time = time.time()
    print("\nEpoch training time: " + str(end_time-start_time) + " seconds")
    for i in range(len(images_test)):
        if classify(weights_matrices, images_test[i]) == label_test[i]:
            score = score+1
    print("Test metric: " + str((score*100)/len(images_test)) + " %\n")
    learning_rate = learning_rate/2

