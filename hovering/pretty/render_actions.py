import os
import re
import glob
import numpy as np
from PIL import Image as PIL_Image
from tqdm import tqdm
import open3d as o3d
from open3d.visualization import rendering

from lib.base_type import ColmapModel
from hovering.helper import (
    get_o3d_pcd, Helper
)


epic_root = '/media/skynet/DATA/Datasets/epic-100/rgb/' #


EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


class PrettyHoverRunner:

    fov = None
    lookat = None
    front = None
    up = None

    epic_img_x0 = 800
    epic_img_y0 = 0
    background_color = [1, 1, 1, 2]  # white;  [1, 1, 1, 0] for black

    def __init__(self, 
                 out_size: str = 'big'):
        if out_size == 'big':
            out_size = (1920, 1080)
        else:
            out_size = (640, 480)
        self.render = rendering.OffscreenRenderer(*out_size)

    def setup(self,
              model_path: str,
              frames_root: str,
              out_dir: str,
              scene_transform=None,
              pcd_model_path=None):
        """
        Args:
            model_path: 
                e.g. '/build_kitchens_3dmap/projects/ahmad/P34_104/sparse_model/'
            frames_root: 
                e.g. '/home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
            out_dir:
                e.g. 'P34_104_out'
            scene_transform: function
        """
        if pcd_model_path == None:
            pcd_model_path = model_path
        # self.model = ColmapModel(model_path)
        pcd_model = ColmapModel(pcd_model_path)
        self.frames_root = frames_root
        self.out_dir = out_dir
        self.scene_transform = scene_transform
        pcd = get_o3d_pcd(pcd_model)

        self.transformed_pcd = self.scene_transform(pcd)
    
    def add_geometry(self, name, geometry, color_arr,
                     apply_scene_transfrom=True):
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = color_arr
        material.point_size = 1.0
        if apply_scene_transfrom:
            geometry = self.scene_transform(geometry)
        self.render.scene.add_geometry(name, geometry, material)

    def test_single_frame(self, 
                          psize,
                          img_id: int =None,
                          clear_geometry: bool = True,):
        """
        Args:
            psize: point size,
                probing a good point size is a bit tricky but very important!
        """
        pcd = self.transformed_pcd

        if clear_geometry:
            self.render.scene.clear_geometry()

        # Get materials
        helper = Helper(point_size=psize)
        white = helper.material('white')
        red = helper.material('red')
        self.helper = helper

        # put on pcd
        self.render.scene.add_geometry('pcd', pcd, white)

        # # put on frustum
        # if img_id is None:
        #     test_img = self.model.get_image_by_id(self.model.ordered_image_ids[0])
        # else:
        #     test_img = self.model.get_image_by_id(img_id)
        # frustum = get_pointer(
        #     sz=0.6, line_radius=0.02, 
        #     colmap_image=test_img)
        # frustum = self.scene_transform(frustum)
        # self.render.scene.add_geometry('frustum', frustum, red)

        self.render.scene.set_background(self.background_color)
        self.render.setup_camera(
            self.fov, self.lookat, self.front, self.up)
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        return img
    
    def render_to_image(self):
        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        return img
        
    
class RunP01_04(PrettyHoverRunner):

    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P01_04_low/'
    pcd_model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_cloud/P01_04/dense'
    # frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P01/P01_04'
    frames_root = f'{epic_root}/P01/P01_04'
    out_dir = './P01_04'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 2.5

    fov = 25
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p01_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*15/180, 180*np.pi/180, 30 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([-2, -2, -3.3])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP01_04.p01_transform,pcd_model_path=RunP01_04.pcd_model_path)

class RunP26_112(PrettyHoverRunner):
    """ Failed """

    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P26_112_low/'
    pcd_model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_cloud/P26_112/dense'
    frames_root = f'{epic_root}/P26/P26_112'
    out_dir = './P26_112'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 2.5

    #fov = 23
    fov=28
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p26_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*30/180, 160*np.pi/180, -20 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([0.3, -2.25, 0])
        #g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1.8, -3, 0])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP26_112.p26_transform,
            pcd_model_path=RunP26_112.pcd_model_path)


class RunP04_01(PrettyHoverRunner):

    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P04_01_low/'
    pcd_model_path = './projects/colmap_models_cloud/P04_01'
    frames_root = f'{epic_root}/P04/P04_01'
    out_dir = './P04_01'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 2.5

    #fov = 23
    fov=28
    lookat = [0, 0, 0]
    front = [-12, 12, 20]
    up = [0, 0, 1]
    def p04_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*30/180, 160*np.pi/180, -20 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([0.3, -2.25, 0])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP04_01.p04_transform,
            pcd_model_path=RunP04_01.pcd_model_path)


class RunP02_109(PrettyHoverRunner):
    model_path = None # '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/registered_models/P02_109_low/'
    pcd_model_path = './projects/colmap_models_cloud/P02_109_dense'
    frames_root = f'{epic_root}/P02/P02_109'
    out_dir = './P02_109'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    #fov = 23
    fov=25
    lookat = [0, 0, 0]
    front = [12, 12, 20]
    up = [0, 0, 1]
    def p02_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*15/180, 180*np.pi/180, -30 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([0.3, -2.25, 0])
        #g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1.8, -3, 0])
        return g

    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP02_109.p02_transform,
            pcd_model_path=RunP02_109.pcd_model_path)

class RunP34_104(PrettyHoverRunner):
    model_path = './projects/colmap_models_cloud/P34_104_dense_v1'
    frames_root = f'{epic_root}/P34/P34_104'  # /home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
    out_dir = './P34_104'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    fov = 19
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p34_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*15/180, 180*np.pi/180, 30 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([-2, -2, -3.3])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP34_104.p34_transform)

class RunP03_117(PrettyHoverRunner):
    model_path = '/home/skynet/Zhifan/epic_fields_full/colmap_models_cloud_barry/P03_117/dense'
    frames_root = f'{epic_root}/P03/P03_117'  # /home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
    out_dir = './P03_117'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    fov = 18
    lookat = [0, 0, 0]
    front = [3, 3, 50]
    up = [0, 0, 1]
    def p03_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*30/180, 160*np.pi/180, -20 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([0.3, -2.25, 0])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP03_117.p03_transform)