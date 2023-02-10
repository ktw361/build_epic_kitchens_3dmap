from typing import Tuple
from functools import cached_property
import tqdm
import numpy as np
import cv2
from PIL import Image
from colmap_converter import colmap_utils

from line_check.line import Line
from line_check.functions import (
    project_line_image, point_line_distance)


class LineChecker:
    
    def __init__(self,
                 model_dir: str,
                 anno_points: np.ndarray):

        def _as_list(path, func):
            return func(path)
            as_list = lambda x: list(dict(x).values())
            return as_list(func(path))

        cameras = _as_list(
            f'{model_dir}/cameras.bin', colmap_utils.read_cameras_binary)
        assert len(cameras) == 1, "Only support single camera"
        self.camera = cameras[1]
        self.points = _as_list(
            f'{model_dir}/points3D.bin', colmap_utils.read_points3d_binary)
        self.images = _as_list(
            f'{model_dir}/images.bin', colmap_utils.read_images_binary)

        self.line = Line(anno_points)

        self._pts_status = None  # point status, dict, True if inside
        self._radius = None
    
    @property
    def example_data(self):
        ki = list(self.images.keys())[0]
        img = self.images[ki]
        kp = list(self.points.keys())[0]
        point = self.points[kp]
        return img, point
    
    @cached_property
    def ordered_image_ids(self):
        return sorted(self.images.keys(), key=lambda x: self.images[x].name)
    
    @property
    def num_points(self):
        return len(self.points)
    
    @property
    def num_images(self):
        return len(self.images)
    
    @property
    def ordered_images(self):
        return [self.images[i] for i in self.ordered_image_ids]
    
    def get_image_by_id(self, image_id: int):
        return self.images[image_id]

    def aggregate(self,
                  radius: float,
                  return_dict: bool = True,
                  debug=False):
        """ Mark all points 
        Args:
            radius: threshold for checking inside
        
        Returns:
            dict, {point_id: bool}
        """
        query_points = np.stack([v.xyz for v in self.points.values()], 0) # (N, 3)
        inds = self.line.check_points(query_points, radius)
        id_array = np.asarray(list(self.points.keys()))  # (N,)
        if not return_dict:
            raise NotImplementedError
        if debug:
            print("Number of points inside: %d / %d" % (inds.sum(), self.num_points))
        self._pts_status = dict(zip(id_array, inds.tolist()))
        self._radius = radius
        return self._pts_status

    def _draw_points2d(self, 
                       img: np.ndarray, 
                       xys: np.ndarray,
                       *args, **kwargs) -> np.ndarray:
        """ Show image from path
        Args:
            img: np.ndarray, (H, W, 3)
            xys: [N, 2] keypoints
        """
        for x, y in xys:
            img = cv2.circle(
                img, (int(x), int(y)), *args, **kwargs)
        return img

    def highlight_2d_points(self, 
                            image_id: int, 
                            display=('inside', 'outside', 'others')):
        """ Highlight 2d retrieved points
        See also visualize_compare()

        Circle for -1 points (no match in 3D )
        Blue for 3d-matched but not inside
        Red for 3d-matched and inside

        Args:
            idx: int, camera pose index 
        """
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        image = self.images[image_id]
        img_name = image.name
        img_path = img_name
        img = np.asarray(Image.open(img_path))

        inside, outside, others = [], [], []
        for i, pid in enumerate(image.point3D_ids):
            if pid == -1:
                others.append(i)
            elif self._pts_status[pid] == True:
                inside.append(i)
            else:
                outside.append(i)
        inside = image.xys[inside]
        outside = image.xys[outside]
        others = image.xys[others]
        if 'others' in display:
            img = self._draw_points2d(img, others, radius=2, color=BLUE, lineType=-1, thickness=1)
        if 'outside' in display:
            img = self._draw_points2d(img, outside, radius=1, color=GREEN, lineType=cv2.FILLED, thickness=2)
        if 'inside' in display:
            img = self._draw_points2d(img, inside, radius=2, color=RED, lineType=cv2.FILLED, thickness=4)
        return img

    def visualize_compare(self, 
                          image_id: int, 
                          display=('inside', 'outside'),
                          lines=('mid',),
                          debug=False):
        """
        Red color: (prediction)
            Annotated Line projected to the image using Predicted camera pose.
        
        Green color: (GT)
            Fitted 2d line using points retrieved from the 3d line.
            2d keypoints are fixed, hence the 2d line is truth.
        
        Args:
            idx: int, camera pose index 
        """
        img_associated = self.highlight_2d_points(image_id, display=display)
        img = img_associated

        mid, ub, lb = project_line_image(
            self.line, self._radius, self.images[image_id], self.camera,
            debug=debug)
        
        for name, line, thick in zip(lines, [mid, ub, lb], [2, 1, 1]):
            if line is None:
                continue
            img = cv2.line(
                img, np.int32(line[0]), np.int32(line[1]), 
                color=(255, 255, 0), thickness=thick, lineType=cv2.LINE_AA)
        return img
    
    def report_single(self, image_id) -> Tuple:
        """
        Returns:
            (status, error)
            - status: 'NO_POINT', 'NO_LINE', 'COMPUTE'
            - error: -1.0 if status != 'COMPUTE'
                mean d(p, L) / image_size
        """
        NO_POINT = 'NO_POINT'
        NO_LINE = 'NO_LINE'
        COMPUTE = 'COMPUTE'

        image = self.images[image_id]
        inside = []
        for i, pid in enumerate(image.point3D_ids):
            if pid == -1:
                continue
            elif self._pts_status[pid] == True:
                inside.append(i)
        inside = image.xys[inside]
        if len(inside) == 0:
            return NO_POINT, -1.0

        mid, _, _ = project_line_image(
            self.line, self._radius, self.images[image_id], self.camera)
        if mid is None:
            return NO_LINE, -1.0
            
        dists = point_line_distance(inside, mid)
        error = (dists ** 2)
        scale_factor = np.sqrt(self.camera.width ** 2 + self.camera.height ** 2)
        error = error.mean() / scale_factor
        return COMPUTE, error
    
    def report_all(self):
        """ Report the number of points inside the line
        """
        if self._pts_status is None:
            raise ValueError("Please run aggregate() first")
        
        results = []
        for image_id in tqdm.tqdm(self.ordered_image_ids):
            r = self.report_single(image_id)
            results.append(r)
        return results