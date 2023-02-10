import numpy as np
from line_check.functions import line_rectangle_check

def line_rectangle_check_test():
    rect = (0, 0, 456, 256)
    vc = (0, 255)
    dir = (1, 0)
    # vc = (0, 100)
    # dir = (1, 0)
    res = line_rectangle_check(
        vc, dir, rect, debug=True)
    print(res)


if __name__ == '__main__':
    line_rectangle_check_test()