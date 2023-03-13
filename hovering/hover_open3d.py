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

SPARSE_PONIT_SIZE = 3
ENHANCE_POINT_SIZE = 0.5

FRUSTUM_SIZE = 1.0
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


class HoverRunner:

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
              extrinsic: np.ndarray,
              out_dir: str,
              is_enhance=True):
        """
        Args:
            model_path: 
                e.g. '/build_kitchens_3dmap/projects/ahmad/P34_104/sparse_model/'
            frames_root: 
                e.g. '/home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
            out_dir:
                e.g. 'P34_104_out'
        """
        self.model = ColmapModel(model_path)
        self.pcd = get_o3d_pcd(self.model)
        self.frames_root = frames_root
        self.extrinsic = extrinsic
        self.out_dir = out_dir

        # if is_enhance:
        #     psize = SPARSE_PONIT_SIZE
        # else:
        #     psize = ENHANCE_POINT_SIZE
        # self.helper = Helper(point_size=psize)
    
    def test_single_frame(self, 
                          psize,
                          img_id: int =None,
                          cam: o3d.camera.PinholeCameraIntrinsic =None,
                          extrinsic: np.ndarray =None):
        """
        Args:
            psize: point size,
                probing a good point size is a bit tricky but very important!
            extrinsic: np.ndarray, 4x4, if not None, will override the extrinsic
        """
        self.render.scene.clear_geometry()

        # Get materials
        helper = Helper(point_size=psize)
        white = helper.material('white')
        red = helper.material('red')

        # put on pcd
        self.render.scene.add_geometry('pcd', self.pcd, white)

        # put on frustum
        if img_id is None:
            test_img = self.model.get_image_by_id(self.model.ordered_image_ids[0])
        else:
            test_img = self.model.get_image_by_id(img_id)
        frustum = get_frustum(
            sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
            colmap_image=test_img, 
            camera_height=EPIC_HEIGHT, camera_width=EPIC_WIDTH)
        epic_img = read_original(
            test_img, frame_root=self.frames_root)
        self.render.scene.add_geometry('frustum', frustum, red)
        self.render.scene.set_background([1, 1, 1, 2])

        # cam = o3d.camera.PinholeCameraIntrinsic(
        #     width=1920, height=-1, fx=1, fy=1, cx=0, cy=0)
        if not cam:
            cam = o3d.camera.PinholeCameraIntrinsic()
        if extrinsic is not None:
            self.render.setup_camera(cam, extrinsic)
        else:
            self.render.setup_camera(cam, self.extrinsic)

        # render.setup_camera(60.0, [0, 0, 0], [D/2, D, 2*D], [0, 0.2, 1])
        # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
        #                                  75000)
        # render.scene.scene.enable_sun_light(True)
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        img[-EPIC_HEIGHT:, 500:500+EPIC_WIDTH] = epic_img

        return img
    
    def run_all(self):
        model = self.model
        render = self.render
        os.makedirs(self.out_dir, exist_ok=True)
        fmt = os.path.join(self.out_dir, '%010d.png')
        red_m = self.helper.material('red')

        render.scene.remove_geometry('traj')
        render.scene.remove_geometry('frustum')

        traj_len = 6
        pos_history = []
        for img_id in tqdm(self.model.ordered_image_ids[:]):
            
            c_img = model.get_image_by_id(img_id)
            frame_idx = int(re.search('\d{10}', c_img.name)[0])
            line_set = get_frustum(
                sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
                colmap_image=c_img, camera_height=EPIC_HEIGHT, 
                camera_width=EPIC_WIDTH)
            pos_history.append(get_cam_pos(c_img))
            
            if len(pos_history) > 2:
                traj = get_trajectory(
                    pos_history, num_line=traj_len, 
                    line_radius=TRAJECTORY_LINE_RADIUS)
                render.scene.add_geometry('traj', traj, red_m)    
            render.scene.add_geometry('frustum', line_set, red_m)
            img = render.render_to_image()
            img = np.asarray(img)
            
            ii = read_original(c_img, frame_root=self.frames_root)
            img[-EPIC_HEIGHT:, 500:500+EPIC_WIDTH] = ii
            
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
        super().__init__()
        self.setup(
            model_path=self.model_path,
            frames_root=self.frames_root,
            out_dir=self.out_dir,
            extrinsic=self.extrinsic,
            is_enhance=self.is_enhance)