from typing import List
import os
import re
import numpy as np
from PIL import Image as PIL_Image
import open3d as o3d
import copy

from hovering.helper import (
    Helper
)
from hovering.hover_open3d_with_interpolation import HoverRunner


epic_root = '/media/skynet/DATA/Datasets/epic-100/rgb/' #
EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


class HandHoverRunner(HoverRunner):

    def render_to_image(self, as_pil=False):
        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        if as_pil:
            return PIL_Image.fromarray(img)
        return img
    
    def reset_canvas(self, with_pcd=True):
        pcd = self.transformed_pcd

        self.render.scene.clear_geometry()

        # Get materials
        helper = Helper(point_size=self.point_size)
        white = helper.material('white')
        self.helper = helper

        # put on pcd
        if with_pcd:
            self.render.scene.add_geometry('pcd', pcd, white)
        self.render.scene.set_background(self.background_color)

        # self.render.setup_camera(self.fov, self.lookat, self.front, self.up)
        self.render.scene.scene.set_sun_light(
            [0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
        self.render.scene.scene.enable_sun_light(True)
        self.render.scene.show_axes(False)
    
    def add_geometry(self, name, geometry, material_color=None, shader='defaultUnlit'):
        if self.render.scene.has_geometry(name):
            self.render.scene.remove_geometry(name)
        material = self.helper.material(material_color, shader=shader)
        geometry = self.scene_transform(copy.deepcopy(geometry))
        self.render.scene.add_geometry(name, geometry, material)
    

class RunP28_101(HandHoverRunner):
    model_path = '/media/skynet/DATA/Zhifan/colmap_projects/epic_fields_data/P28_101.json'
    pcd_model_path = '/home/barry/Ahmad/colmap_epic_fields/colmap_models_cloud/P28_101/dense.ply'
    view_path = './json_files/hover_viewpoints/P28_101_hand1.json'

    vid = re.search('P\d{2}_\d{2,3}', model_path)[0]
    kid = vid[:3]
    frames_root = f'/media/skynet/DATA/Datasets/epic-100/rgb/{kid}/{vid}/'
    out_dir = f'./{vid}'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 3.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup(
            self.model_path,
            viewstatus_path=self.view_path,
            frames_root=self.frames_root,
            out_dir=self.out_dir,
            scene_transform=lambda x: x,
            pcd_model_path=self.pcd_model_path)

        
if __name__ == '__main__':
    pass