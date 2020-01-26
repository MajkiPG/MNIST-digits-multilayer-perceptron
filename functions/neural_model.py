import numpy as np
import math

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    output_copy = np.copy(x)
    output_copy[output_copy<0]=0
    return output_copy

def tanh(x):
    return np.tanh(x)

def compute_layer_output_activation(input_signal, weights):
    output = weights.dot(input_signal)
    return sigmoid(output)

def compute_layer_output(input_signal, weights):
    return weights.dot(input_signal)

def compute_outputs(model, input_signal):
    outputs = []

    outputs.append(compute_layer_output_activation(input_signal, model[0]))
    for i in range(1, len(model)):
        if i != len(model)-1:
            outputs.append(compute_layer_output_activation(outputs[i-1], model[i]))
        else:
            outputs.append(compute_layer_output_activation(outputs[i-1], model[i]))
    outputs.append(compute_layer_output(outputs[0], model[1]))
    outputs[-1] = softmax(outputs[-1])
    return outputs

def classify(model, input_signal):
    outputs = compute_outputs(model, input_signal)
    return(np.argmax(outputs[-1]))

