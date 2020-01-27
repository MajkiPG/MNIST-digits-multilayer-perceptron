import numpy as np

def compute_correction_vector_output(desired_output, output):
    """
    Computes correction vector for output layer.
    @params:
        desired_output  - desired output of neural network
        output          - output of neural network
    @returns:
        correction_vector   - correction vector for output layer weights matrix
    """
    outputs_difference = desired_output - output
    output_inversion = 1 - output
    return outputs_difference*output_inversion*output

def compute_correction_vector_hidden(child_weight_matrix, child_correction_vector, output):
    """
    Computes correction vector for hidden layer.
    @params:
        child_weight_matrix     - weights matrix of child layer
        child_correction_vector - correction vector of child layer weights matrix
        output                  - output of current layer
    @returns:
        correction_vector       - correction vector for current layer weights matrix
    """
    weight_matrix_transformed = np.transpose(child_weight_matrix).dot(child_correction_vector)
    output_inversion = 1 - output
    return weight_matrix_transformed*output_inversion*output

def backpropagate(weights_matrices, outputs, input, desired_output, learning_rate=0.1):
    """
    Performs backpropagation basing on learning sample and its desired output.
    @params:
        weights_matrices - Required : list of models weight matrices
        outputs          - Required : list of all layers outputs based on learning sample
        input            - Required : learning sample
        desired_output   - Required : desired output
        learning_rate    - Optional : learning rate (default = 0.1)
    @returns:
        weight_matrices  - corrected list of models weight matrices after backpropagation
    """
    correction_vectors = []
    for i in range(len(weights_matrices)-1, -1, -1):
        if i == len(weights_matrices)-1:
            correction_vectors.append(compute_correction_vector_output(desired_output, outputs[i]))
            correction_matrix = learning_rate * (np.outer(correction_vectors[-1], outputs[i-1]))
            weights_matrices[i] += correction_matrix
        else:
            correction_vectors.append(compute_correction_vector_hidden(weights_matrices[i+1], correction_vectors[-1], outputs[i]))
            if i == 0:
                correction_matrix = learning_rate * (np.outer(correction_vectors[-1], input))
            else:
                correction_matrix = learning_rate * (np.outer(correction_vectors[-1], outputs[i-1]))
            weights_matrices[i] += correction_matrix
    return weights_matrices