import numpy as np
from main import dot_matrix_with_splits


# 3x3 matrix
x = [[12, 7, 3],
     [4, 5, 6],
     [7, 8, 9]]

# 3x4 matrix (3 rows by 4 columns)
y = [[5, 8, 1, 2],
     [6, 7, 3, 0],
     [4, 5, 9, 1]]


# 3x4 matrix
expected_result = [[114, 160, 60, 27],
                   [74, 97, 73, 14],
                   [119, 157, 112, 23]]


def test(function):
    function()
    print('----------', function.__name__, "passed")


@test
def test_numpy_dot():
    x_matrix = np.matrix(x)
    y_matrix = np.matrix(y)
    matrix_dot = x_matrix * y_matrix

    actual_result = np.squeeze(np.asarray(matrix_dot)).tolist()
    assert actual_result == expected_result


@test
def test_dot_matrix_with_splits_by_two():
    whole_dot = dot_matrix_with_splits(x, y, number_of_splits=2)

    actual_result = np.squeeze(np.asarray(whole_dot)).tolist()
    assert actual_result == expected_result


@test
def test_dot_matrix_with_splits_by_three():
    whole_dot = dot_matrix_with_splits(x, y, number_of_splits=3)

    actual_result = np.squeeze(np.asarray(whole_dot)).tolist()
    assert actual_result == expected_result


print('All tests passed!')
