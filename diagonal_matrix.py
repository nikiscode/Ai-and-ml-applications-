import numpy as np

def upend_with_loop(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][i] = 1.0

    return matrix
result_loop_matrix = upend_with_loop(3)
print("Diagonal matrix generated using a for loop:")
display(result_loop_matrix
