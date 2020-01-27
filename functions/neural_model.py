import numpy as np
#import math

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    """Compute sigmoid activation values for each sets of scores in x."""
    return 1/(1+np.exp(-x))

def compute_layer_output_activation(input_signal, weights):
    """
    Computes output of given layer with sigmoid activation function.
    @params:
        input_signal    - input signal
        weights         - weights matrix of current layer
    @returns:
        output  - output signal of current layer
    """
    output = weights.dot(input_signal)
    return sigmoid(output)

def compute_layer_output(input_signal, weights):
    """
    Computes output of given layer.
    @params:
        input_signal    - input signal
        weights         - weights matrix of current layer
    @returns:
        output  - output signal of current layer
    """
    return weights.dot(input_signal)

def compute_outputs(model, input_signal):
    """
    Computes outputs of all neural network layers. Activation function is sigmoid. Performs softmax normalization on resultant output.
    @params:
        model           - list of models weight matrices
        input_signal    - input signal
    @returns:
        outputs   - list of all layers outputs (last signal in list is networks output)
    """
    outputs = []
    for i in range(len(model)):
        if i == 0:
            outputs.append(compute_layer_output_activation(input_signal, model[i]))
        elif i == len(model)-1:
            outputs.append(compute_layer_output(outputs[-1], model[i]))
        else:
            outputs.append(compute_layer_output_activation(outputs[-1], model[i]))
    outputs[-1] = softmax(outputs[-1])
    return outputs

def classify(model, input_signal):
    """
    Computes digit classification result.
    @params:
        model           - list of models weight matrices
        input_signal    - input signal
    @returns:
        digit   - 0-9 detected integer 
    """
    outputs = compute_outputs(model, input_signal)
    return(np.argmax(outputs[-1]))

