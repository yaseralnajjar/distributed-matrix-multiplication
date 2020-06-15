import numpy as np


def dot_matrix_with_splits(x, y, number_of_splits):

    def chunks(lst, n):
        # https://stackoverflow.com/a/312464/4565520
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

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
    dots = []
    for sub_matrix in sub_matrixes:
        dots.append(np.matrix(sub_matrix) * y_matrix)

    result = np.concatenate(dots)
    return result
