import numpy as np
from line_check.line import line_fit


def line_fit_test():
    points = np.array([
        [0, 0, 0],
        [10, 0, 0],
        # noise
        [1, 0.2, -0.8],
        [5, 0.1, -0.3],
        [11, -1, 1]
        ])
    v0, v1 = line_fit(points)
    return v0, v1

if __name__ == '__main__':
    line_fit_test()