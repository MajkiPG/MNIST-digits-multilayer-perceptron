import numpy as np

def compute_correction_vector_output(desired_output, output):
    outputs_difference = desired_output - output
    output_inversion = 1 - output
    return outputs_difference*output_inversion*output

def compute_correction_vector_hidden(older_weight_matrix, older_correction_vector, output):
    weight_matrix_transformed = np.transpose(older_weight_matrix).dot(older_correction_vector)
    output_inversion = 1 - output
    return weight_matrix_transformed*output_inversion*output

def backpropagate(weights_matrices, outputs, desired_output):
    correction_vectors = []
    correction_vectors.append(compute_correction_vector_output(desired_output, outputs[-1]))
    cv_counter = 0
    for num in range(-2, -1-len(weights_matrices), -1):
        correction_vectors.append(compute_correction_vector_hidden(weights_matrices[num+1], correction_vectors[cv_counter], outputs[num]))
        cv_counter = cv_counter+1
    return correction_vectors