import numpy as np


large_x = np.random.randint(2, 9, (10, 10))
large_y = np.random.randint(2, 9, (10, 10))
large_x_dot_y_expected_result = np.matrix(large_x) * np.matrix(large_y)

print(large_x)

np.savetxt('large_matrix_x.txt', large_x, fmt='%d')
np.savetxt('large_matrix_y.txt', large_y, fmt='%d')
np.savetxt('large_matrix_result.txt', large_x_dot_y_expected_result, fmt='%d')
