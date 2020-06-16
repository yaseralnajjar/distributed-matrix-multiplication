import os
from multiprocessing import Process
import numpy as np


def chunks(lst, n):
    # src: https://stackoverflow.com/a/312464/4565520
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def dot_matrix_with_splits(x, y, number_of_splits):
    """
    2 chunks
    ONLY x will be chunked
    y will be passed as it is for both dots
    1x3 * 3x4 = 1x4
    2x3 * 3x4 = 2x4
    1x4 concatenates 2x4 = 3x4
    """

    sub_matrixes = chunks(x, number_of_splits)

    y_matrix = np.matrix(y)
    for sub_matrix in sub_matrixes:
        yield np.matrix(sub_matrix) * y_matrix


def dot_matrix_map_and_reduce(x, y, number_of_splits):
    y_matrix = np.matrix(y)

    working_directory = os.path.dirname(os.path.realpath(__file__))

    def map_to_file(sub_matrix, host_number):
        result_dot = np.matrix(sub_matrix) * y_matrix
        result_file = os.path.join(working_directory, f'host{host_number}', 'result')
        np.save(result_file, result_dot)

    def reduce_result(number_of_splits):
        """
        read mapped files and concatenate them
        write reduced result to one file
        """
        matrixes = []
        for host_number in range(number_of_splits):
            mapped_file = os.path.join(working_directory, f'host{host_number}', 'result.npy')
            matrixes.append(np.load(mapped_file))

        reduced_file = os.path.join(working_directory, f'reduced', 'result')
        np.save(reduced_file, np.concatenate(matrixes))

    sub_matrixes = chunks(x, number_of_splits)
    processes = []
    for index, sub_matrix in enumerate(sub_matrixes):
        map_to_file(sub_matrix, index)
        #Process(target=map_to_file, args=(sub_matrix, index))

    # kick them off
    for process in processes:
        process.start()

    # now wait for them to finish
    for process in processes:
        process.join()

    reduce_result(number_of_splits)
    reduced_result_file = os.path.join(working_directory, f'reduced', 'result.npy')
    result_matrix = np.load(reduced_result_file)

    return result_matrix
