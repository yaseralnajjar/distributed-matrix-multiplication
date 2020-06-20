import os
from math import ceil
from multiprocessing import Process
import numpy as np


def chunks(lst, n):
    # src: https://stackoverflow.com/a/312464/4565520
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def dot_matrix_with_splits(x, y, number_of_splits):
    """
    THIS FUNCTION IS WRITTEN ONLY TO TEST SPLITTING MATRIX DOT OPERATION
    2 chunks
    ONLY x will be chunked
    y will be passed as it is for both dots
    1x3 * 3x4 = 1x4
    2x3 * 3x4 = 2x4
    1x4 concatenates 2x4 = 3x4
    """

    chunk_size = ceil(len(x) / number_of_splits)
    sub_matrixes = chunks(x, chunk_size)

    y_matrix = np.matrix(y)
    for sub_matrix in sub_matrixes:
        yield np.matrix(sub_matrix) * y_matrix


def map_to_file(sub_matrix, y_matrix, host_number):
    working_directory = os.path.dirname(os.path.realpath(__file__))
    result_file = os.path.join(working_directory, f'host{host_number}', 'result')

    result_dot = np.matrix(sub_matrix) * y_matrix
    np.save(result_file, result_dot)


def dot_matrix_map(x, y, number_of_splits):
    y_matrix = np.matrix(y)
    chunk_size = ceil(len(x) / number_of_splits)
    sub_matrixes = chunks(x, chunk_size)

    processes = []
    for index, sub_matrix in enumerate(sub_matrixes):
        # map_to_file(sub_matrix, y_matrix, index)  # run synchronously
        new_process = Process(target=map_to_file, args=(sub_matrix, y_matrix, index))
        processes.append(new_process)

    # kick them off
    for process in processes:
        process.start()

    # now wait for them to finish
    for process in processes:
        process.join()


def dot_matrix_reduce(x, y, number_of_splits):
    working_directory = os.path.dirname(os.path.realpath(__file__))

    def reduce_result(number_of_splits):
        """
        read mapped files and concatenate them
        write reduced result to one file
        """
        matrixes = []
        for host_number in range(number_of_splits):
            mapped_file = os.path.join(working_directory, f'host{host_number}', 'result.npy')
            matrixes.append(np.load(mapped_file))

        reduced_file = os.path.join(working_directory, 'reduced', 'result')
        np.save(reduced_file, np.concatenate(matrixes))

    reduce_result(number_of_splits)
    reduced_result_file = os.path.join(working_directory, 'reduced', 'result.npy')

    return reduced_result_file


def dot_matrix_map_and_reduce(x, y, number_of_splits):
    dot_matrix_map(x, y, number_of_splits)
    reduced_result_file = dot_matrix_reduce(x, y, number_of_splits)

    result_matrix = np.load(reduced_result_file)
    return result_matrix
