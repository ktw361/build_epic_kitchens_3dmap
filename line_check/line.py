import numpy as np


def line_fit(points: np.ndarray) -> np.ndarray:
    """ Least-Square fitting of a line
    Args:
        points: (N, 3) array of points
    
    Returns:
        vc: (3,) array of center point
        dir: (3,) array of direction
    """
    avg = np.mean(points, 0)
    centered = points - avg
    np.linalg.svd(centered)

    uu, dd, vv = np.linalg.svd(centered)
    dir = vv[0]
    dir = dir / np.linalg.norm(dir)
    return avg, dir


class Line:
    """ An infinite 3D line to denote Annotated Line """
    
    def __init__(self, 
                 anno_points: np.ndarray):
        """
        Args:
            anno_points: (N, 3)
                points annotated using some GUI, denoting points along the ground truth line
        """
        vc, dir = line_fit(anno_points)
        self.vc = vc
        self.dir = dir
        self.v0 = vc
        self.v1 = vc + dir
    
    def __repr__(self) -> str:
        return f'vc: {str(self.vc)} \ndir: {str(self.dir)}'
    
    def check_single_point(self, 
                           point: np.ndarray,
                           radius: float) -> bool:
        """
        point-to-line = (|(p-v_0)x(p-v_1)|)/(|v_1 - v_0|)

        Args:
            point: (3,) array of point
            radius: threshold for checking inside
        """
        area2 = np.linalg.norm(np.cross(point - self.v0, point - self.v1))
        base_len = np.linalg.norm(self.v1 - self.v0)
        d = area2 / base_len
        return True if d < radius else False

    def check_points(self, 
                     points: np.ndarray,
                     diameter: float) -> np.ndarray:
        """
        Args:
            points: (N, 3) array of points
            diameter: threshold for checking inside
        
        Returns:
            (N,) bool array
        """
        area2 = np.linalg.norm(np.cross(points - self.v0, points - self.v1), axis=1)
        base_len = np.linalg.norm(self.v1 - self.v0)
        d = area2 / base_len
        return d < diameter
        # return np.where(d < diameter)[0]