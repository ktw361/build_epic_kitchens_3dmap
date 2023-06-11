import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix

from line_check.line import Line


def line_rectangle_check(cen, dir, rect,
                         eps=1e-6,
                         debug=False):
    """
    Args:
        cen, dir: (2,) float
        rect: Tuple (xmin, ymin, xmax, ymax)

    Returns:
        num_intersect: int
        inters: (num_intersect, 2) float
    """
    x1, y1 = cen
    u1, v1 = dir
    xmin, ymin, xmax, ymax = rect
    rect_loop = np.asarray([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
        [xmin, ymin]
    ], dtype=np.float32)
    x2, y2 = rect_loop[:4, 0], rect_loop[:4, 1]
    u2 = rect_loop[1:, 0] - rect_loop[:-1, 0]
    v2 = rect_loop[1:, 1] - rect_loop[:-1, 1]

    t2 = (v1*x1 - u1*y1) - (v1*x2 - u1*y2)
    divisor = (v1*u2 - v2*u1)
    cond = np.abs(divisor) > eps

    t2[~cond] = -1
    t2[cond] = t2[cond] / divisor[cond]

    keep = (t2 >= 0) & (t2 <= 1)
    num_intersect = np.sum(keep)
    uv = np.stack([u2, v2], 1)
    inters = rect_loop[:4, :] + t2[:, None] * uv
    inters = inters[keep, :]
    if debug:
        print(cond)
        print(t2)
        print(rect_loop[:4, :])
        print(uv)
        """
        t2: (4,) float
            -1 if not intersect
            [0, 1] if intersect
        """
    return num_intersect, inters


def project_line_image(line: Line,
                       radius: float,
                       image,
                       camera,
                       debug=False):
    """ Project three lines: center, upper bound, lower bound

    Args:
        line:
            -vc: (3,) float
            -dir: (3,) float
        image:
            -qvec: (4,) world to cam
            -tvec: (3,) world to cam
        camera:
            -width,
            -height
            -params (8,) fx, fy, cx, cy, k1, k2, p1, p2
                typical value
                    2.48599189e+02,  2.49046265e+02,  2.28000000e+02,  1.28000000e+02,
                    -2.32192380e-02,  5.96217313e-03, -1.31808242e-03,  1.88818864e-03

    Returns:
        (st, ed), (st_ub, ed_ub), (st_lb, ed_lb): (2,) float
    """
    cen, dir = line.vc, line.dir
    qvec, tvec = image.qvec, image.tvec
    # Represent as column vector
    cen, dir = transform_line(cen, dir, qvec, tvec)
    width, height = camera.width, camera.height
    fx, fy, cx, cy, k1, k2, p1, p2 = camera.params

    cen_uv = cen[:2] / cen[2]
    cen_uv = cen_uv * np.array([fx, fy]) + np.array([cx, cy])
    dir_uv = ((dir + cen)[:2] / (dir + cen)[2]) - (cen[:2] / cen[2])
    dir_uv = dir_uv * np.array([fx, fy])
    dir_uv = dir_uv / np.linalg.norm(dir_uv)
    # Previous wrong implementation:
    # dir_uv = dir[:2] / dir[2]
    # dir_uv = dir_uv * np.array([fx, fy])
    # dir_uv = dir_uv / np.linalg.norm(dir_uv)

    # TODO Assume fx = fy
    # TODO: distort
    thickness = radius / cen[2] * fx
    normal = np.array([-dir_uv[1], dir_uv[0]])
    ub_uv = cen_uv + normal * thickness
    lb_uv = cen_uv - normal * thickness

    cen_line = None
    num_inters, inters = line_rectangle_check(
        cen_uv, dir_uv, (0, 0, width, height), debug=False)
    if num_inters == 2:
        cen_line = (inters[0], inters[1])

    ub_line = None
    num_inters, inters = line_rectangle_check(
        ub_uv, dir_uv, (0, 0, width, height), debug=False)
    if num_inters == 2:
        ub_line = (inters[0], inters[1])

    lb_line = None
    num_inters, inters = line_rectangle_check(
        lb_uv, dir_uv, (0, 0, width, height), debug=False)
    if num_inters == 2:
        lb_line = (inters[0], inters[1])

    if debug:
        print('cen', cen)
        print('ub', ub_line)
        print('lb', lb_line)

    return cen_line, ub_line, lb_line


def transform_line(cen, dir, qvec, tvec):
    """ Transform line from world to camera coordinate
    Returns:
        cen, dir: (3,) float
    """
    rot_w2c = quaternion_to_matrix(torch.as_tensor(qvec))
    cen = rot_w2c @ cen + tvec
    dir = rot_w2c @ dir
    return cen.numpy(), dir.numpy()


def point_line_distance(points, line):
    """ Compute distance between points and line
    Args:
        points: (N, 2) float
        line:
            -st: (2,) float
            -ed: (2,) float
    Returns:
        dist: (N,) float
    """
    st, ed = line
    dist = np.cross(ed - st, st - points) / np.linalg.norm(ed - st)
    return dist
