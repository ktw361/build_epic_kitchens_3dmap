from argparse import ArgumentParser
from typing import List
import os
import os.path as osp
import re
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d
from lib.base_type import JsonColmapModel
from libzhifan import io
try:
    import ujson as json
except ImportError:
    import json
from open3d.visualization import rendering
from registration.transform_utils import transform_images

from hovering.helper import (Helper, get_frustum, read_original,
    get_cam_pos, get_trajectory, set_offscreen_as_gui
)

from moviepy import editor
from PIL import ImageDraw, ImageFont

EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile', help='path to transforms.json')
    parser.add_argument('--pcd_model_path', type=str, default=None)
    parser.add_argument('--json_data_root', default='projects/epic_fields_data')
    parser.add_argument('--view_path', type=str, required=True,
                        help='path to the view file, e.g. json_files/hover_viewoints/P22_115_view1.json')
    args = parser.parse_args()
    return args


class HoverRunner:

    fov = None
    lookat = None
    front = None
    up = None

    epic_img_x0 = 0
    epic_img_y0 = 0
    background_color = [1, 1, 1, 2]  # white;  [1, 1, 1, 0] for black

    def __init__(self, out_size: str = 'big'):
        if out_size == 'big':
            out_size = (1920, 1080)
        else:
            out_size = (640, 480)
        self.render = rendering.OffscreenRenderer(*out_size)

    def setup(self,
              model: JsonColmapModel,
              viewstatus_path: str,
              frames_root: str,
              out_dir: str,
              pcd_model_path=None):
        """
        Args:
            model_path:
                e.g. '/build_kitchens_3dmap/projects/ahmad/P34_104/sparse_model/' TODO: update this
            viewstatus_path:
                path to viewstatus.json, CTRL-c output from Open3D gui
            frames_root:
                e.g. '/home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
            out_dir:
                e.g. 'P34_104_out'
        """
        self.model = model
        # if model_path.endswith('.json'):
            # self.model = JsonColmapModel(model_path)
        # else:
            # self.model = ColmapModel(model_path)
        if pcd_model_path == None:
            points = self.model['points']
            pcd_np = [v[:3] for v in points]
            pcd_rgb = [np.asarray(v[3:6]) / 255 for v in points]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
            pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
        else:
            assert pcd_model_path.endswith('.ply')
            pcd = o3d.io.read_point_cloud(pcd_model_path)

        self.viewstatus_path = viewstatus_path
        self.frames_root = frames_root
        self.out_dir = out_dir
        self.transformed_pcd = pcd  # The `transformed` name is legacy, it's not actually being tranfromed.
    
    def resetup(self, model, frames_root, out_dir):
        self.model = model
        self.frames_root = frames_root
        self.out_dir = out_dir

    def test_single_frame(self,
                          psize,
                          img_id: int =None,
                          clear_geometry: bool =True,
                          lay_image: bool =True,
                          sun_light: bool =False,
                          show_first_frustum: bool =True):
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
        with open(self.viewstatus_path) as f:
            viewstatus = json.load(f)
        set_offscreen_as_gui(self.render, viewstatus)

        # put on frustum
        if img_id is None:
            test_img = self.model.get_image_by_id(self.model.ordered_image_ids[0])
        else:
            test_img = self.model.get_image_by_id(img_id)
        frustum = get_frustum(
            sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS,
            colmap_image=test_img,
            camera_height=EPIC_HEIGHT, camera_width=EPIC_WIDTH)
        if show_first_frustum:
            self.render.scene.add_geometry('first_frustum', frustum, red)
        self.render.scene.set_background(self.background_color)

        if sun_light:
            self.render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                            75000)
            self.render.scene.scene.enable_sun_light(True)
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        if lay_image:
            epic_img = read_original(
                test_img, frame_root=self.frames_root)
            img[self.epic_img_y0:self.epic_img_y0+EPIC_HEIGHT,
                self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = epic_img
        return img

    def run_all(self, step_size: int =1, fps: int =30):
        model = self.model
        render = self.render
        os.makedirs(self.out_dir, exist_ok=True)
        fmt = os.path.join(self.out_dir, '%010d.jpg')
        red_m = self.helper.material('red')
        white_m = self.helper.material('white')

        render.scene.remove_geometry('frustum')

        traj_len = 20
        pos_history = []
        for img_id in tqdm(self.model.ordered_image_ids[::step_size]):

            c_img = model.get_image_by_id(img_id)
            frame_idx = int(re.search('\d{10}', c_img.name)[0])
            frustum = get_frustum(
                sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS,
                colmap_image=c_img, camera_height=EPIC_HEIGHT,
                camera_width=EPIC_WIDTH)
            pos_history.append(get_cam_pos(c_img))

            if len(pos_history) > 2:
                traj = get_trajectory(
                    pos_history, num_line=traj_len,
                    line_radius=TRAJECTORY_LINE_RADIUS)
                if render.scene.has_geometry('traj'):
                    render.scene.remove_geometry('traj')
                render.scene.add_geometry('traj', traj, white_m)
                # for i, line in enumerate(lines):
                #     geom_name = f'line_{i}'
                #     if render.scene.has_geometry(geom_name):
                #         render.scene.remove_geometry(geom_name)
                #     render.scene.add_geometry(f'line_{i}', line, white_m)
            render.scene.add_geometry('frustum', frustum, red_m)
            img = render.render_to_image()
            img = np.asarray(img)

            ii = read_original(c_img, frame_root=self.frames_root)
            img[self.epic_img_y0:self.epic_img_y0+EPIC_HEIGHT, 
                self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = ii

            # capture the screen and convert to PIL image
            img_pil = Image.fromarray(img)
            I1 = ImageDraw.Draw(img_pil)
            myFont = ImageFont.truetype('FreeMono.ttf', 65)
            I1.text((500, 1000), c_img.name.split('.')[0], font=myFont, fill =(0, 0, 0))
            # bbox2 = (1450,823, 1906, 1079)
            # I1.rectangle(bbox2, outline='red', width=5)
            img_pil.save(fmt % frame_idx)

            render.scene.remove_geometry('frustum')

        # Gen output
        video_fps = fps
        print("Generating video...")
        seq = sorted(glob.glob(os.path.join(self.out_dir, '*.jpg')))
        clip = editor.ImageSequenceClip(seq, fps=video_fps)
        clip.write_videofile(os.path.join(self.out_dir, f'out-step{step_size}-fps{fps}.mp4'))
    

if __name__ == '__main__':
    args = parse_args()
    scene = osp.basename(args.infile).replace('.json', '')  # e.g. P01AB
    kid = scene[:3]
    identity_transform = lambda x: x

    # vid = re.search('P\d{2}_\d{2,3}', args.model)[0]
    runner = HoverRunner()
    runner.setup(
        model=None,
        viewstatus_path=args.view_path,
        frames_root=None,
        out_dir=None,
        pcd_model_path=args.pcd_model_path)

    transforms = io.read_json(args.infile)
    for vid in sorted(transforms.keys()):
        scale = transforms[vid]['scale']
        rot = np.asarray(transforms[vid]['rot']).reshape(3, 3)
        transl = np.asarray(transforms[vid]['transl'])
        frames_root = f'/media/skynet/DATA/Datasets/epic-100/rgb/{kid}/{vid}/'
        assert os.path.exists(frames_root)
        out_dir = f'outputs/regitration/{scene}/{vid}'

        json_model_path = f'{args.json_data_root}/{vid}.json'
        assert osp.exists(json_model_path)
        json_data = io.read_json(json_model_path)
        image_arr = np.asarray([im for im in json_data['images'].values()])
        tfed_image_arr = transform_images(
            image_arr, scale, rot, transl)
        for k, im in zip(json_data['images'].keys(), tfed_image_arr):
            json_data['images'][k] = im.tolist()
        json_model = JsonColmapModel(json_data)

        runner.resetup(json_model, frames_root, out_dir)
        runner.test_single_frame(0.2, show_first_frustum=False)
        print(f"Running {vid}")
        runner.run_all(step_size=3, fps=20)
