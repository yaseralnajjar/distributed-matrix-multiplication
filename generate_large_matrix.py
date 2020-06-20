import numpy as np


SIZE = 1000

large_x = np.random.randint(2, high=9, size=(SIZE, SIZE))
large_y = np.random.randint(2, high=9, size=(SIZE, SIZE))
large_x_dot_y_expected_result = np.matrix(large_x) * np.matrix(large_y)


np.savetxt('large_matrix_x.txt', large_x, fmt='%d')
np.savetxt('large_matrix_y.txt', large_y, fmt='%d')
np.savetxt('large_matrix_result.txt', large_x_dot_y_expected_result, fmt='%d')
