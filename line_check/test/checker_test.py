import numpy as np
from line_check.line import line_fit, Line
from line_check.checker import LineChecker
from line_check.functions import transform_line, project_line_image, point_line_distance


def main():
    anno_points = [
        4.63313, -0.2672, 2.55641,
        -5.22596, 0.352575, 3.04684,
        0.675789, -0.0019428, 2.77022
    ]

    anno_points = np.asarray(anno_points).reshape(-1, 3)

    image_id = 2292
    image_id = 3079
    checker = LineChecker('projects/stables_single/P01_01-homo/sparse/0',
                      anno_points=anno_points)

    radius = 0.2 # 0.1
    _ = checker.aggregate(radius=radius, debug=True)

    r = checker.report_single(image_id)
    print(r)

if __name__ == '__main__':
    main()