from typing import List
import os
import re
import glob
import numpy as np
from PIL import Image as PIL_Image
from tqdm import tqdm
import open3d as o3d
from open3d.visualization import rendering
import copy

from lib.base_type import ColmapModel
from hovering.helper import (
    get_o3d_pcd, Helper, get_frustum,get_frustum_fixed,get_frustum_green, read_original,
    get_cam_pos, get_trajectory
)
from hovering.pretty.utils import get_pretty_trajectory
from colmap_converter.colmap_utils import (
    BaseImage, Point3D, Camera, Image
)

from moviepy import editor
from PIL import ImageDraw, ImageFont

epic_root = '/media/skynet/DATA/Datasets/epic-100/rgb/' #
EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


class HandHoverRunner:

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
        self.model = ColmapModel(model_path)
        pcd_model = ColmapModel(pcd_model_path)
        self.frames_root = frames_root
        self.out_dir = out_dir
        self.scene_transform = scene_transform
        self.pcd = get_o3d_pcd(pcd_model)

        self.transformed_pcd = self.scene_transform(copy.deepcopy(self.pcd))
    
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

        self.render.setup_camera(self.fov, self.lookat, self.front, self.up)
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
    
    def test_single_frame(self, 
                          psize,
                          img_id: int =None,
                          clear_geometry=True):
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

        # put on frustum
        if img_id is None:
            test_img = self.model.get_image_by_id(self.model.ordered_image_ids[0])
        else:
            test_img = self.model.get_image_by_id(img_id)
        frustum = get_frustum(
            sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
            colmap_image=test_img, 
            camera_height=EPIC_HEIGHT, camera_width=EPIC_WIDTH)
        frustum = self.scene_transform(frustum)
        epic_img = read_original(
            test_img, frame_root=self.frames_root)
        self.render.scene.add_geometry('frustum', frustum, red)
        self.render.scene.set_background(self.background_color)

        self.render.setup_camera(
            self.fov, self.lookat, self.front, self.up)
        # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
        #                                  75000)
        # render.scene.scene.enable_sun_light(True)
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        img[self.epic_img_y0:self.epic_img_y0+EPIC_HEIGHT, 
            self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = epic_img
        # import PIL
        # PIL.Image.fromarray(img).save('a.png')
        return img

    def run_all(self):
        model = self.model
        render = self.render
        os.makedirs(self.out_dir, exist_ok=True)
        fmt = os.path.join(self.out_dir, '%010d.jpg')
        red_m = self.helper.material('red')
        green_m = self.helper.material('blue')

        render.scene.remove_geometry('traj')
        render.scene.remove_geometry('frustum')
        render.scene.remove_geometry('frustum2')

        traj_len = 25
        pos_history = []
        for img_id in tqdm(self.model.ordered_image_ids[:]):
            
            c_img = model.get_image_by_id(img_id)
            frame_idx = int(re.search('\d{10}', c_img.name)[0])
            frustum = get_frustum(
                sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
                colmap_image=c_img, camera_height=EPIC_HEIGHT, 
                camera_width=EPIC_WIDTH)
            frustum = self.scene_transform(frustum)
            pos_history.append(get_cam_pos(c_img))
            
            frustum_fixed = get_frustum_fixed(
                sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
                colmap_image=c_img, camera_height=EPIC_HEIGHT, 
                camera_width=EPIC_WIDTH)
                
            frustum_fixed = self.scene_transform(frustum_fixed)
            
            if len(pos_history) > 2:
                traj = get_trajectory(
                    pos_history, num_line=traj_len, 
                    line_radius=TRAJECTORY_LINE_RADIUS)
                traj = self.scene_transform(traj)
                render.scene.add_geometry('traj', traj, red_m)    
            render.scene.add_geometry('frustum', frustum, red_m)
            ####render.scene.add_geometry('frustum2', frustum_fixed, green_m)
            img = render.render_to_image()
            img = np.asarray(img)
            
            ek_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P11/P11_03'
            ii = read_original(c_img, frame_root=self.frames_root)

            img[-EPIC_HEIGHT-1:-1, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = ii #ii_ek

            m = Image.fromarray(img)
            draw = ImageDraw.Draw(m)
            bbox2 = (1450,823, 1906, 1079)
            draw.rectangle(bbox2, outline='red', width=5)
            m.save(fmt % frame_idx)

            render.scene.remove_geometry('traj')
            render.scene.remove_geometry('frustum')
            render.scene.remove_geometry('frustum2')

        # Gen output
        video_fps = 12
        print("Generating video...")
        seq = sorted(glob.glob(os.path.join(self.out_dir, '*.jpg')))
        clip = editor.ImageSequenceClip(seq, fps=video_fps)
        clip.write_videofile(os.path.join(self.out_dir, 'out.mp4'))
    
    def singel_frame_with_front(self, 
                                front, 
                                img_id=None,
                                traj: list=None):
        """
        Args:
            traj: transformed
        """
        pcd = self.transformed_pcd

        self.render.scene.clear_geometry()

        # Get materials
        psize = self.point_size
        helper = Helper(point_size=psize)
        white = helper.material('white')
        red = helper.material('red')

        # put on pcd
        self.render.scene.add_geometry('pcd', pcd, white)

        # put on frustum
        if img_id is None:
            test_img = self.model.get_image_by_id(self.model.ordered_image_ids[0])
        else:
            test_img = self.model.get_image_by_id(img_id)
        frustum = get_frustum(
            sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
            colmap_image=test_img, 
            camera_height=EPIC_HEIGHT, camera_width=EPIC_WIDTH)
        frustum = self.scene_transform(frustum)
        self.render.scene.add_geometry('frustum', frustum, red)
        if traj:
            self.render.scene.add_geometry('traj', traj, red)
        self.render.scene.set_background(self.background_color)

        self.render.setup_camera(
            self.fov, self.lookat, front, self.up)
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        epic_img = read_original(
            test_img, frame_root=self.frames_root)
        img[-EPIC_HEIGHT:, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = epic_img

        return img

    def run_circulating(self, num_loop=2, Radius=14.142, Height=20):
        num_imgs = len(self.model.ordered_image_ids)
        theta = np.linspace(0, num_loop*2*np.pi, num_imgs)
        fronts = [
            [Radius*np.cos(t), Radius*np.sin(t), Height] for t in theta
        ]

        model = self.model
        render = self.render
        os.makedirs(self.out_dir, exist_ok=True)
        fmt = os.path.join(self.out_dir, '%010d.png')

        render.scene.remove_geometry('traj')
        render.scene.remove_geometry('frustum')

        traj_len = 8
        pos_history = []
        for front, img_id in tqdm(
            zip(fronts, self.model.ordered_image_ids),
            total=len(fronts)):
            
            c_img = model.get_image_by_id(img_id)
            frame_idx = int(re.search('\d{10}', c_img.name)[0])
            pos_history.append(get_cam_pos(c_img))
            
            if len(pos_history) >= 2:
                traj = get_trajectory(
                    pos_history, num_line=traj_len, 
                    line_radius=TRAJECTORY_LINE_RADIUS)
                traj = self.scene_transform(traj)
            else:
                traj = None
            
            img = self.singel_frame_with_front(
                front=front, img_id=img_id, traj=traj)
            img = np.asarray(img)
            
            ii = read_original(c_img, frame_root=self.frames_root)
            img[-EPIC_HEIGHT:, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = ii
            
            Image.fromarray(img).save(fmt % frame_idx)

        # Gen output
        video_fps = 12
        print("Generating video...")
        seq = sorted(glob.glob(os.path.join(self.out_dir, '*.png')))
        clip = editor.ImageSequenceClip(seq, fps=video_fps)
        clip.write_videofile(os.path.join(self.out_dir, 'out.mp4'))


class RunP28_101(HandHoverRunner):
    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P28_101_low/'
    pcd_model_path = '/home/barry/Ahmad/colmap_epic_fields/colmap_models_cloud/P28_101/dense'
    frames_root = '/media/skynet/DATA/Datasets/epic-100/rgb/P28/P28_101'
    out_dir = './P28_101'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 3.5

    # Global
    # fov=30
    # lookat = [0, 0, 0]
    # front = [5, 10, 15]
    fov=6
    lookat = [-4, -3, 0]
    front = [0, 15, 20]
    up = [0, 0, 1]
    def p28_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*30/180, 160*np.pi/180, 20 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([-3.8, 0.25, 0])
        #g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1.8, -3, 0])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP28_101.p28_transform,
            pcd_model_path=RunP28_101.pcd_model_path)


class RunP22_115(HandHoverRunner):
    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P22_115_low/'
    pcd_model_path = '/home/barry/Ahmad/colmap_epic_fields/colmap_models_cloud/P22_115/dense'
    frames_root = '/media/skynet/DATA/Datasets/epic-100/rgb/P22/P22_115/'
    out_dir = './P22_115'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 3.5

    # Global
    fov=15
    lookat = [0, 0, 0]
    front = [0, 0, 20] # fov=6
    # lookat = [-4, -3, 0]
    # front = [0, 15, 20]
    up = [0, 0, -1]
    def p22_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*30/180, 160*np.pi/180, 20 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([0, 0, 0])
        #g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1.8, -3, 0])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP22_115.p22_transform,
            pcd_model_path=RunP22_115.pcd_model_path)


class RunP09_104(HandHoverRunner):
    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P09_104_low/'
    pcd_model_path = '/home/barry/Ahmad/colmap_epic_fields/colmap_models_cloud/P09_104/dense'
    frames_root = '/media/skynet/DATA/Datasets/epic-100/rgb/P09/P09_104/'
    out_dir = './P09_104'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 3.5

    # Global
    fov=30
    lookat = [0, 0, 0]
    front = [-10, 0, 30]
    # fov=6
    # lookat = [-4, -3, 0]
    # front = [0, 15, 20]
    up = [0, 0, 1]
    def p09_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*30/180, 160*np.pi/180, 20 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([0, 0, 0])
        #g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1.8, -3, 0])
        return g
    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP09_104.p09_transform,
            pcd_model_path=RunP09_104.pcd_model_path)

        
if __name__ == '__main__':
    pass