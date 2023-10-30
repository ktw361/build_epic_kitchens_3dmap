from typing import List
import os
import re
import glob
import numpy as np
from PIL import Image as PIL_Image

from lib.base_type import ColmapModel
from hovering.helper import (
    Helper, get_frustum, read_original,
    get_cam_pos, get_pretty_trajectory,
)
from hovering.hover_open3d_with_interpolation import HoverRunner
from colmap_converter.colmap_utils import (
    BaseImage, Point3D, Camera, Image
)


epic_root = '/media/skynet/DATA/Datasets/epic-100/rgb/' #
EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


class PrettyHoverRunner(HoverRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def render_trajectory(self, cimgs: List[Image], with_pcd=True,
                          traj_line_radius=TRAJECTORY_LINE_RADIUS,
                          traj_darkness=1.0, 
                          manual_frustum_cimgs=None,
                          point_size=2.5):
        """ cimgs: ordered 
        0: current , 1: previous, 2: previous of previous

        Args:
            cimgs: display coloured trajectory connecting these cimgs
            manual_frustum_cimgs: display frustums for these cimgs
        """
        render = self.render
        helper = Helper(point_size=point_size)
        red_m = helper.material('red')
        white_m = helper.material('white')

        self.render.scene.clear_geometry()
        if with_pcd:
            self.render.scene.add_geometry('pcd', self.transformed_pcd, white_m)

        pos_history = []
        for cimg in cimgs:
            pos_history.append(get_cam_pos(cimg))
        
        for idx, cimg in enumerate(manual_frustum_cimgs):
            frustum = get_frustum(
                sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
                colmap_image=cimg, camera_height=EPIC_HEIGHT, 
                camera_width=EPIC_WIDTH)
            frustum = self.scene_transform(frustum)
            render.scene.add_geometry(f'frustum_{idx}', frustum, red_m)

        if len(pos_history) > 2:
            lines = get_pretty_trajectory(
                pos_history, num_line=len(pos_history), 
                line_radius=traj_line_radius,
                darkness=traj_darkness)
            for i, line in enumerate(lines):
                line = self.scene_transform(line)
                render.scene.add_geometry(f'line_{i}', line, white_m)    
        img = render.render_to_image()
        img = np.asarray(img)
        return img