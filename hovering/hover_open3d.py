import os
import re
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d
from open3d.visualization import rendering

from lib.base_type import ColmapModel
from hovering.helper import (
    get_o3d_pcd, Helper, get_frustum, read_original,
    get_cam_pos, get_trajectory
)
from moviepy import editor


EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 1.0
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


class HoverRunner:

    fov = None
    lookat = None
    front = None
    up = None

    epic_img_x0 = 500
    epic_img_y0 = -EPIC_HEIGHT
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
              scene_transform=None):
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
        self.model = ColmapModel(model_path)
        self.frames_root = frames_root
        self.out_dir = out_dir
        self.scene_transform = scene_transform
        pcd = get_o3d_pcd(self.model)
        self.transformed_pcd = self.scene_transform(pcd)

    def test_single_frame(self, 
                          psize,
                          img_id: int =None):
        """
        Args:
            psize: point size,
                probing a good point size is a bit tricky but very important!
        """
        pcd = self.transformed_pcd

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

        return img
    
    def run_all(self):
        model = self.model
        render = self.render
        os.makedirs(self.out_dir, exist_ok=True)
        fmt = os.path.join(self.out_dir, '%010d.png')
        red_m = self.helper.material('red')

        render.scene.remove_geometry('traj')
        render.scene.remove_geometry('frustum')

        traj_len = 8
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
            
            if len(pos_history) > 2:
                traj = get_trajectory(
                    pos_history, num_line=traj_len, 
                    line_radius=TRAJECTORY_LINE_RADIUS)
                traj = self.scene_transform(traj)
                render.scene.add_geometry('traj', traj, red_m)    
            render.scene.add_geometry('frustum', frustum, red_m)
            img = render.render_to_image()
            img = np.asarray(img)
            
            ii = read_original(c_img, frame_root=self.frames_root)
            img[-EPIC_HEIGHT:, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = ii
            
            Image.fromarray(img).save(fmt % frame_idx)
            # o3d.io.write_image(fmt % frame_idx, img, 9)

            render.scene.remove_geometry('traj')
            render.scene.remove_geometry('frustum')

        # Gen output
        video_fps = 12
        print("Generating video...")
        seq = sorted(glob.glob(os.path.join(self.out_dir, '*.png')))
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



class RunP34_104_sparse(HoverRunner):

    model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P34_104/sparse_model/'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
    extrinsic = np.float32([
        [-1, 0, 0, -10],
        [0, -1, 0, -8],
        [0, 0, 1, 18],
        [0, 0, 0, 1]
    ])
    out_dir = './P34_104_vis'
    is_enhance = False

    def __init__(self):
        raise ValueError("Deprecated!")
        super().__init__()
        self.setup(
            model_path=self.model_path,
            frames_root=self.frames_root,
            out_dir=self.out_dir,
            extrinsic=self.extrinsic,
            is_enhance=self.is_enhance)
        

class RunP02_109(HoverRunner):

    model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P02_109/enhanced_model/'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P02/P02_109'
    out_dir = './P02_109'

    epic_img_x0 = 1000
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.0

    fov = 30
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p02_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*15/180, 180*np.pi/180, -30 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1, -3, 0])
        return g

    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP02_109.p02_transform)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--version', type=str)
    args = parser.parse_args()

    if args.version == 'P02_white':
        runner = RunP02_109()
        runner.out_dir = 'outputs/hover_P02_109_white'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all()
    elif args.version == 'P02_black':
        # TODO: set traj color to white
        runner = RunP02_109()
        runner.out_dir = 'outputs/hover_P02_109_black'
        runner.background_color = [1, 1, 1, 0]
        runner.test_single_frame(runner.point_size)
        runner.run_all()
    elif args.version == 'P02_circle':
        runner = RunP02_109()
        runner.out_dir = 'outputs/hover_P02_109_circle'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_circulating(
            num_loop=2, Radius=16, Height=23)