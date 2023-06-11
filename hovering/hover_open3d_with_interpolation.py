from typing import List
import os
import re
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d
try:
    import ujson as json
except ImportError:
    import json
from open3d.visualization import rendering

from colmap_converter.colmap_utils import Image as ColmapImage
from hovering.helper import (
    Helper, 
    get_frustum, get_frustum_fixed, get_frustum_green, 
    read_original,
    get_cam_pos, get_trajectory, get_pretty_trajectory, set_offscreen_as_gui
)

from moviepy import editor
from PIL import ImageDraw, ImageFont

EPIC_WIDTH = 456
EPIC_HEIGHT = 256

FRUSTUM_SIZE = 0.6
FRUSTUM_LINE_RADIUS = 0.02

TRAJECTORY_LINE_RADIUS = 0.02


class JsonColmapModel:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path) as f:
            model = json.load(f)
        self.camera = model['camera']
        self.points = model['points']
        self.images = sorted(model['images'], key=lambda x: x[-1])  # qw, qx, qy, qz, tx, ty, tz, frame_name
    
    @property
    def ordered_image_ids(self):
        return list(range(len(self.images)))
    
    @property
    def ordered_images(self) -> List[ColmapImage]:
        return [self.get_image_by_id(i) for i in self.ordered_image_ids]
    
    def get_image_by_id(self, image_id: int) -> ColmapImage:
        img_info = self.images[image_id]
        cimg = ColmapImage(
            id=image_id, qvec=img_info[:4], tvec=img_info[4:7], camera_id=0, 
            name=img_info[7], xys=[], point3D_ids=[])
        return cimg


def get_interpolation(point1=np.array([1, 2, 3]),point2=np.array([5, 7, 10]),N=10):
    # Define the two points as 3D numpy arrays
    interpolated_points = []
    interpolated_points.append(point1) # include the first point in the interpolation

    # Calculate the distance between the two points
    distance = np.linalg.norm(point2 - point1)

    # Calculate the direction of the line connecting the two points
    direction = (point2 - point1) / distance

    # Calculate the positions of the N interpolated points along the line
    interpolated_points.extend([point1 + i * direction * distance / (N + 1) for i in range(1, N + 1)])

    # Print the interpolated points
    return interpolated_points

def slerp(q1, q2, N):
    # Ensure that the input quaternions are unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate the dot product of the quaternions
    dot = np.dot(q1, q2)

    # If the dot product is negative, flip the second quaternion
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Set the interpolation parameters
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    t = np.linspace(0, 1, N+2)[1:-1]

    # Interpolate between the quaternions
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    q_interp = np.outer(w1, q1) + np.outer(w2, q2)

    # Normalize the interpolated quaternions and return them as a list of arrays
    result = [q1]
    result.extend([q / np.linalg.norm(q) for q in q_interp])
    return result


class HoverRunner:

    fov = None
    lookat = None
    front = None
    up = None

    epic_img_x0 = 800
    epic_img_y0 = 0
    background_color = [1, 1, 1, 2]  # white;  [1, 1, 1, 0] for black

    def __init__(self, out_size: str = 'big'):
        if out_size == 'big':
            out_size = (1920, 1080)
        else:
            out_size = (640, 480)
        self.render = rendering.OffscreenRenderer(*out_size)

    def setup(self,
              model_path: str,
              viewstatus_path: str,
              frames_root: str,
              out_dir: str,
              scene_transform=None,
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
            scene_transform: function
        """
        self.model = JsonColmapModel(model_path)
        if pcd_model_path == None:
            pcd_model_path = model_path
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
        self.scene_transform = scene_transform
        self.transformed_pcd = self.scene_transform(pcd)
    
    def test_single_frame(self, 
                          psize,
                          img_id: int =None,
                          clear_geometry: bool =True,
                          lay_image: bool =True):
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
        frustum = self.scene_transform(frustum)
        self.render.scene.add_geometry('frustum', frustum, red)
        self.render.scene.set_background(self.background_color)

        # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
        #                                  75000)
        # render.scene.scene.enable_sun_light(True)
        self.render.scene.show_axes(False)

        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        if lay_image:
            epic_img = read_original(
                test_img, frame_root=self.frames_root)
            img[self.epic_img_y0:self.epic_img_y0+EPIC_HEIGHT, 
                self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = epic_img
        return img
    
    def run_all_int(self):
        """ run_all() with interpolation """
        model = self.model
        render = self.render
        os.makedirs(self.out_dir, exist_ok=True)
        fmt = os.path.join(self.out_dir, '%010d.jpg')
        red_m = self.helper.material('red')
        green_m = self.helper.material('green')

        render.scene.remove_geometry('traj')
        render.scene.remove_geometry('frustum')
        render.scene.remove_geometry('frustum2')

        traj_len = 25
        pos_history = []
        for i in tqdm(range(0,len(self.model.ordered_image_ids[:])-1)):
            img_id1 = self.model.ordered_image_ids[:][i]
            img_id2 = self.model.ordered_image_ids[:][i+1]

            c_img1 = model.get_image_by_id(img_id1)
            c_img2 = model.get_image_by_id(img_id2)

            img1_num = int(c_img1.name[-14:-4])
            img2_num = int(c_img2.name[-14:-4])

            if img1_num > 0 and img1_num < 1000000:

                if abs(img2_num - img1_num) > 1:
                    interpolations_t = get_interpolation(c_img1.tvec,c_img2.tvec,N=abs(img2_num - img1_num)-1)
                    interpolations_q = slerp(c_img1.qvec,c_img2.qvec,N=abs(img2_num - img1_num)-1)#get_interpolation(c_img1.qvec,c_img2.qvec,N=abs(img2_num - img1_num)-1)
                else:
                    interpolations_t = [c_img1.tvec]
                    interpolations_q = [c_img1.qvec]

                for ti in range(0,len(interpolations_t)):
                    c='red'
                    t = interpolations_t[ti]
                    q = interpolations_q[ti]
                    c_img = Image(tvec=t,qvec=q,id=0,camera_id=c_img1.camera_id,name='frame_%010d.jpg'%(img1_num+ti) ,xys=[],point3D_ids=[])
                    # print(c_img.name)
                    frame_idx = int(re.search('\d{10}', c_img.name)[0])
                    frustum = get_frustum(
                        sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
                        colmap_image=c_img, camera_height=EPIC_HEIGHT, 
                        camera_width=EPIC_WIDTH)
                    frustum = self.scene_transform(frustum)
                    frustum_fixed = get_frustum_green(
                        sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
                        colmap_image=c_img, camera_height=EPIC_HEIGHT, 
                        camera_width=EPIC_WIDTH)
                    frustum_fixed = self.scene_transform(frustum_fixed)

                    pos_history.append(get_cam_pos(c_img))
                    
                    if len(pos_history) > 2:
                        traj = get_trajectory(
                            pos_history, num_line=traj_len, 
                            line_radius=TRAJECTORY_LINE_RADIUS)
                        traj = self.scene_transform(traj)
                        render.scene.add_geometry('traj', traj, red_m)
                    if  ti > 0:
                        render.scene.add_geometry('frustum2', frustum_fixed, green_m)
                        c='green'  
                    else:
                        render.scene.add_geometry('frustum', frustum, red_m)
                        c='red'  
                    img = render.render_to_image()
                    img = np.asarray(img)
                    
                    ii = read_original(c_img, frame_root=self.frames_root)
                    img[-EPIC_HEIGHT-1:-1, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = ii
                    
                    img_pil = Image.fromarray(img)
                    # Call draw Method to add 2D graphics in an image
                    I1 = ImageDraw.Draw(img_pil)
                    
                    # Custom font style and font size
                    myFont = ImageFont.truetype('FreeMono.ttf', 65)
                    
                    # Add Text to an image
                    I1.text((500, 1000), c_img.name.split('.')[0], font=myFont, fill =(0, 0, 0))
                    # define the bbox coordinates (left, top, right, bottom)
                    bbox2 = (1450,823, 1906, 1079)
                    # draw the bbox
                    #draw.rectangle(bbox, outline='red', width=5)
                    I1.rectangle(bbox2, outline=c, width=5)
                            
                    img_pil.save(fmt % frame_idx)
                    # o3d.io.write_image(fmt % frame_idx, img, 9)

                    render.scene.remove_geometry('traj')
                    render.scene.remove_geometry('frustum')
                    render.scene.remove_geometry('frustum2')
        video_fps = 12
        print("Generating video...")
        seq = sorted(glob.glob(os.path.join(self.out_dir, '*.jpg')))
        clip = editor.ImageSequenceClip(seq, fps=video_fps)
        clip.write_videofile(os.path.join(self.out_dir, 'out.mp4'))

    def run_all(self):
        model = self.model
        render = self.render
        os.makedirs(self.out_dir, exist_ok=True)
        fmt = os.path.join(self.out_dir, '%010d.jpg')
        red_m = self.helper.material('red')
        white_m = self.helper.material('white')

        render.scene.remove_geometry('frustum')
        render.scene.remove_geometry('frustum2')

        traj_len = 40
        pos_history = []
        for img_id in tqdm(self.model.ordered_image_ids[::10]):
            
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
                lines = get_pretty_trajectory(
                    pos_history, num_line=traj_len, 
                    line_radius=0.08,
                    darkness=0.4)
                for i, line in enumerate(lines):
                    line = self.scene_transform(line)
                    geom_name = f'line_{i}'
                    if render.scene.has_geometry(geom_name):
                        render.scene.remove_geometry(geom_name)
                    render.scene.add_geometry(f'line_{i}', line, white_m)
            render.scene.add_geometry('frustum', frustum, red_m)
            ####render.scene.add_geometry('frustum2', frustum_fixed, green_m)
            img = render.render_to_image()
            img = np.asarray(img)
            
            ii = read_original(c_img, frame_root=self.frames_root)
            img[-EPIC_HEIGHT-1:-1, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = ii
            
            # capture the screen and convert to PIL image
            img_pil = Image.fromarray(img)
            I1 = ImageDraw.Draw(img_pil)
            myFont = ImageFont.truetype('FreeMono.ttf', 65)
            I1.text((500, 1000), c_img.name.split('.')[0], font=myFont, fill =(0, 0, 0))
            bbox2 = (1450,823, 1906, 1079)
            I1.rectangle(bbox2, outline='red', width=5)
            img_pil.save(fmt % frame_idx)

            render.scene.remove_geometry('traj')
            render.scene.remove_geometry('frustum')
            render.scene.remove_geometry('frustum2')

        # Gen output
        video_fps = 20
        print("Generating video...")
        seq = sorted(glob.glob(os.path.join(self.out_dir, '*.jpg')))
        clip = editor.ImageSequenceClip(seq, fps=video_fps)
        clip.write_videofile(os.path.join(self.out_dir, 'out.mp4'))
    
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


class RunP09_104(HoverRunner):
    # model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P09_104_low/'
    model_path = '/home/skynet/Zhifan/build_epic_kitchens_3dmap/projects/json_models/P09_104_skeletons_extend.json'
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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--pcd_model_path', type=str, default=None)
    parser.add_argument('--view_path', type=str, required=True, 
                        help='path to the view file, e.g. json_files/hover_viewoints/P22_115_view1.json')
    args = parser.parse_args()

    identity_transform = lambda x: x
    vid = re.search('P\d{2}_\d{2,3}', args.model)[0]
    kid = vid[:3]
    frames_root = f'/media/skynet/DATA/Datasets/epic-100/rgb/{kid}/{vid}/'
    runner = HoverRunner()
    runner.setup(
        model_path=args.model, 
        viewstatus_path=args.view_path,
        frames_root=frames_root,
        out_dir=f'outputs/hover_{vid}',
        scene_transform=identity_transform,
        pcd_model_path=args.pcd_model_path)
    runner.epic_img_x0 = 0
    runner.epic_img_y0 = 0
    runner.test_single_frame(3.5) # runner.point_size)
    runner.run_all()