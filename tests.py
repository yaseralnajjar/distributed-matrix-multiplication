import os
import numpy as np
from timer import timer
from main import dot_matrix_with_splits, dot_matrix_map_and_reduce


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

working_directory = os.path.dirname(os.path.realpath(__file__))


def convert_np_to_list(np_matrix):
    return np.squeeze(np.asarray(np_matrix)).tolist()


def test(function):
    if __name__ == '__main__':
        tests_to_run = [
            # 'test_large_matrix_with_splits_by_three',
            # 'test_large_matrix_map_and_reduce_by_three',
        ]
        if tests_to_run:
            if function.__name__ in tests_to_run:
                function()
                print('----------', function.__name__, 'passed')
        else:
            function()
            print('----------', function.__name__, 'passed')


@test
@timer('    Took: ', precision=9)
def test_numpy_dot():
    x_matrix = np.matrix(x)
    y_matrix = np.matrix(y)
    matrix_dot = x_matrix * y_matrix

    actual_result = convert_np_to_list(matrix_dot)
    assert actual_result == expected_result


@test
@timer('    Took: ', precision=9)
def test_dot_matrix_with_splits_by_two():
    dots_generator = dot_matrix_with_splits(x, y, number_of_splits=2)
    dots = list(dots_generator)
    whole_dot = np.concatenate(dots)

    actual_result = np.squeeze(np.asarray(whole_dot)).tolist()
    assert actual_result == expected_result


@test
@timer('    Took: ', precision=9)
def test_dot_matrix_with_splits_by_three():
    dots_generator = dot_matrix_with_splits(x, y, number_of_splits=3)
    dots = list(dots_generator)
    whole_dot = np.concatenate(dots)

    actual_result = convert_np_to_list(whole_dot)
    assert actual_result == expected_result


def recreate_dirs(folders):
    def rmtree(directory):
        # src: https://stackoverflow.com/a/52324968/4565520
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(directory)

    for folder in folders:
        folder_to_recreate = os.path.join(working_directory, folder)
        try:
            rmtree(folder_to_recreate)
        except Exception as e:
            # print(e)
            pass
        os.mkdir(folder_to_recreate)


@test
@timer('    Took: ', precision=9)
def test_dot_matrix_map_and_reduce_by_two():
    recreate_dirs(['host0', 'host1', 'reduced'])

    whole_dot = dot_matrix_map_and_reduce(x, y, number_of_splits=2)
    actual_result = convert_np_to_list(whole_dot)
    assert actual_result == expected_result


@test
@timer('    Took: ', precision=9)
def test_dot_matrix_map_and_reduce_by_three():
    recreate_dirs(['host0', 'host1', 'host2', 'reduced'])

    whole_dot = dot_matrix_map_and_reduce(x, y, number_of_splits=3)
    actual_result = convert_np_to_list(whole_dot)
    assert actual_result == expected_result


large_x = convert_np_to_list(np.loadtxt(os.path.join(working_directory, 'large_matrix_x.txt')))
large_y = convert_np_to_list(np.loadtxt(os.path.join(working_directory, 'large_matrix_y.txt')))
large_x_dot_y_expected_result = convert_np_to_list(np.loadtxt(
    os.path.join(working_directory, 'large_matrix_result.txt')))


@test
@timer('    Took: ', precision=9)
def test_large_matrix_with_splits_by_three():
    dots_generator = dot_matrix_with_splits(large_x, large_y, number_of_splits=3)
    dots = list(dots_generator)
    whole_dot = np.concatenate(dots)
    actual_result = convert_np_to_list(whole_dot)

    assert actual_result == large_x_dot_y_expected_result

    # for i, row in enumerate(actual_result):
    #    print(row)
    #    print('qoooooooooq')
    #    print(large_x_dot_y_expected_result[i])
    #    assert row == large_x_dot_y_expected_result[i]
#
    #    for j, inner_row in enumerate(row):
    #        pass
#
    #    print('------------------')


@test
@timer('    Took: ', precision=9)
def test_large_matrix_map_and_reduce_by_three():
    recreate_dirs(['host0', 'host1', 'host2', 'reduced'])

    # whole_dot = dot_matrix_map(large_x, large_y, number_of_splits=2)
    # actual_result = np.squeeze(np.asarray(np.concatenate(list(whole_dot)))).tolist()
    whole_dot = dot_matrix_map_and_reduce(large_x, large_y, number_of_splits=3)
    actual_result = convert_np_to_list(whole_dot)

    assert actual_result == large_x_dot_y_expected_result

    # for i, row in enumerate(actual_result):
    #    print(row)
    #    print('qoooooooooq')
    #    print(large_x_dot_y_expected_result[i])
    #    assert row == large_x_dot_y_expected_result[i]
#
    #    for j, inner_row in enumerate(row):
    #        pass
#
    #    print('------------------')


if __name__ == '__main__':
    print('All tests passed!')
