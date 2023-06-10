from typing import List
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
    get_o3d_pcd, Helper, get_frustum,get_frustum_fixed,get_frustum_green, read_original,
    get_cam_pos, get_trajectory, get_pretty_trajectory,
)
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

def get_interpolation(point1=np.array([1, 2, 3]),point2=np.array([5, 7, 10]),N=10):
    # Define the two points as 3D numpy arrays
    #point1 = np.array([1, 2, 3])
    #point2 = np.array([5, 7, 10])
    interpolated_points = []
    interpolated_points.append(point1) # include the first point in the interpolation

    # Calculate the distance between the two points
    distance = np.linalg.norm(point2 - point1)

    # Calculate the direction of the line connecting the two points
    direction = (point2 - point1) / distance

    # Calculate the positions of the N interpolated points along the line
    interpolated_points.extend([point1 + i * direction * distance / (N + 1) for i in range(1, N + 1)])

    # Print the interpolated points
    #print(interpolated_points)
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
        pcd = get_o3d_pcd(pcd_model)

        self.transformed_pcd = self.scene_transform(pcd)
    
    def render_to_image(self):
        img_buf = self.render.render_to_image()
        img = np.asarray(img_buf)
        return img

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
                    
                    img_pil = PIL_Image.fromarray(img)
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
            ii_ek = read_original(c_img, frame_root=ek_root)

            img[-EPIC_HEIGHT-1:-1, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = ii #ii_ek

            #img[-EPIC_HEIGHT-1:-1, self.epic_img_x0-470:self.epic_img_x0+EPIC_WIDTH-470] = ii
            
            # capture the screen and convert to PIL image
            m = Image.fromarray(img)

            # create a drawing context
            draw = ImageDraw.Draw(m)

            # define the bbox coordinates (left, top, right, bottom)
            bbox = (1450,823, 1906, 1079)

            bbox = (980,823, 1436, 1079)
            bbox2 = (1450,823, 1906, 1079)
            # draw the bbox
            #draw.rectangle(bbox, outline='red', width=5)
            draw.rectangle(bbox2, outline='red', width=5)

            #myFont = ImageFont.truetype('FreeMono.ttf', 55)
            #myFont2 = ImageFont.truetype('FreeMono.ttf', 45)
            #draw.text((1640, 770), "GT", font=myFont,fill=(0, 0, 0),stroke_width=2)
            #draw.text((1040, 770), "Rendered View", font=myFont2,fill=(0, 0, 0),stroke_width=2)


            m.save(fmt % frame_idx)
            ##Image.fromarray(img).save(fmt % frame_idx)
            # o3d.io.write_image(fmt % frame_idx, img, 9)

            render.scene.remove_geometry('traj')
            render.scene.remove_geometry('frustum')
            render.scene.remove_geometry('frustum2')

        # Gen output
        video_fps = 12
        print("Generating video...")
        seq = sorted(glob.glob(os.path.join(self.out_dir, '*.jpg')))
        clip = editor.ImageSequenceClip(seq, fps=video_fps)
        clip.write_videofile(os.path.join(self.out_dir, 'out.mp4'))
    
    def render_trajectory(self, cimgs: List[Image], with_pcd=True,
                          traj_line_radius=TRAJECTORY_LINE_RADIUS,
                          traj_darkness=1.0, 
                          manual_frustum_indices=None):
        """ cimgs: ordered 
        0: current , 1: previous, 2: previous of previous
        """
        render = self.render
        helper = Helper(point_size=2.5)
        red_m = helper.material('red')
        white_m = helper.material('white')

        self.render.scene.clear_geometry()
        if with_pcd:
            self.render.scene.add_geometry('pcd', self.transformed_pcd, white_m)

        cur_cimg = cimgs[-1]
        pos_history = []
        for cimg in cimgs:
            pos_history.append(get_cam_pos(cimg))
        
        if manual_frustum_indices is None:
            frustum = get_frustum(
                sz=FRUSTUM_SIZE, line_radius=FRUSTUM_LINE_RADIUS, 
                colmap_image=cur_cimg, camera_height=EPIC_HEIGHT, 
                camera_width=EPIC_WIDTH)
            frustum = self.scene_transform(frustum)
            render.scene.add_geometry('frustum', frustum, red_m)
        else:
            for idx in manual_frustum_indices:
                cimg = cimgs[idx]
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

        epic_img = read_original(cur_cimg, frame_root=self.frames_root)
        # img[-EPIC_HEIGHT:, self.epic_img_x0:self.epic_img_x0+EPIC_WIDTH] = epic_img
        # img[-EPIC_HEIGHT:, :EPIC_WIDTH] = epic_img
        return img

    
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

    # model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/registered_models/P02_109_low/'
    model_path = '/home/skynet/Zhifan/epic_fields_full/skeletons_extend/P02_109_low/'
    pcd_model_path = '/home/skynet/Zhifan/epic_fields_full/colmap_models_cloud_barry/P02_109/dense'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P02/P02_109'
    out_dir = './P02_109'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    fov=22
    lookat = [0, 0, 0]
    front = [10, 15, 25]
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
        
class RunP34_104(HoverRunner):

    model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P34_104/enhanced_model/'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
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


class RunP01_106(HoverRunner):

    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_runs/P01_106_SIMPLE_PINHOLE/sparse/0'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P01/P01_106'
    out_dir = './P01_106'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    fov = 19
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
            scene_transform=RunP01_106.p01_transform)

class RunP02_109_video1(HoverRunner):


    model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P02_109/enhanced_model/'
    frames_root = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/video1_frames'
    out_dir = './P02_109_video1'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    fov = 23
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p02_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*15/180, 180*np.pi/180, -30 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1.8, -3, 0])
        return g

    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP02_109.p02_transform)
        

class RunP02_109_video2(HoverRunner):


    model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P02_109/enhanced_model/'
    frames_root = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/video2_frames'
    out_dir = './P02_109_video2'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    fov = 23
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p02_transform(g):
        t = - np.float32([0.04346319,1.05888072,2.09330869])
        rot = o3d.geometry.get_rotation_matrix_from_xyz(
            [-np.pi*15/180, 180*np.pi/180, -30 * np.pi / 180])
        g = g.translate(t).rotate(rot, center=(0,0,0)).translate([1.8, -3, 0])
        return g

    def __init__(self):
        super().__init__()
        self.setup(
            self.model_path,
            self.frames_root,
            self.out_dir,
            scene_transform=RunP02_109_video2.p02_transform)

class RunP03_13(HoverRunner):


    #model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P02_109/enhanced_model/'
    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P03_13_low/'
    pcd_model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_cloud/P03_13/dense'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P03/P03_13'
    out_dir = './P03_13'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 2.5

    #fov = 23
    fov=28
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p03_transform(g):
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
            scene_transform=RunP03_13.p03_transform,
            pcd_model_path=RunP03_13.pcd_model_path)

class RunP01_04(HoverRunner):


    #model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P02_109/enhanced_model/'
    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P01_04_low/'
    pcd_model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_cloud/P01_04/dense'
    # frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P01/P01_04'
    frames_root = '/media/skynet/DATA/Datasets/epic-100/rgb/P01/P01_04'
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

class RunP26_112(HoverRunner):


    #model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P02_109/enhanced_model/'
    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P26_112_low/'
    pcd_model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_cloud/P26_112/dense'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P26/P26_112'
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

class RunP11_03(HoverRunner):

    #model_path = '/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P02_109/enhanced_model/'
    model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_registered/P11_03_low/'
    pcd_model_path = '/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/colmap_models_cloud/P11_03/dense'
    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P11/P11_03'
    out_dir = './P11_03'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 2.5

    #fov = 23
    fov=28
    lookat = [0, 0, 0]
    front = [10, 10, 20]
    up = [0, 0, 1]
    def p11_transform(g):
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
            scene_transform=RunP11_03.p11_transform,
            pcd_model_path=RunP11_03.pcd_model_path)
        

class RunP34_104(HoverRunner):
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

class RunP06_102(HoverRunner):
    model_path = '/home/skynet/Zhifan/epic_fields_full/skeletons_extend/P06_102_low/'
    pcd_path = '/home/skynet/Zhifan/epic_fields_full/colmap_models_cloud_barry/P06_102/dense'
    frames_root = f'{epic_root}/P06/P06_102'  # /home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
    out_dir = './P06_102'

    epic_img_x0 = 1450
    background_color = [1, 1, 1, 2]  # white

    point_size = 1.5

    fov = 30
    lookat = [0, 0, 0]
    front = [8, 8, 20]
    up = [0, 0, 1]
    def p06_transform(g):
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
            scene_transform=RunP06_102.p06_transform,
            pcd_model_path=RunP06_102.pcd_path)    

        
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--version', type=str, default='P01_04_white')
    args = parser.parse_args()

    if args.version == 'P01_white':
        runner = RunP01_106()
        runner.out_dir = 'outputs/hover_P01_106_white'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all()
    elif args.version == 'P26_white':
        runner = RunP26_112()
        runner.out_dir = 'outputs/hover_P26_112_white'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all_int()
    elif args.version == 'P11_white':
        runner = RunP11_03()
        runner.out_dir = 'outputs/hover_P11_03_white'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all()

    elif args.version == 'P03_white':
        runner = RunP03_13()
        runner.out_dir = 'outputs/hover_P03_13_white_reg_full'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all_int()
    elif args.version == 'P01_04_white':
        runner = RunP01_04()
        runner.out_dir = 'outputs/hover_P01_04_white_reg_full'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all_int()
    elif args.version == 'P02_white':
        runner = RunP02_109()
        runner.out_dir = 'outputs/hover_P02_109_white_reg_full'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all_int()
    elif args.version == 'P02_white_video1':
        runner = RunP02_109_video1()
        runner.out_dir = 'outputs/hover_P02_109_white_video1'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all()
    
    elif args.version == 'P02_white_video2':
        runner = RunP02_109_video2()
        runner.out_dir = 'outputs/hover_P02_109_white_video2_v2'
        runner.background_color = [1, 1, 1, 2]
        runner.test_single_frame(runner.point_size)
        runner.run_all()
    
    elif args.version == 'P34_white':
        runner = RunP34_104()
        runner.out_dir = 'outputs/hover_P34_104_white'
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
        runner.epic_img_x0 = 0
        runner.epic_img_y0 = 0
        runner.test_single_frame(runner.point_size)
        runner.run_circulating(
            num_loop=2, Radius=16, Height=23)

#ffmpeg -framerate 50 -pattern_type glob -i 'outputs/hover_P02_109_white_reg_full/*.jpg' -c:v libx264 -r 50 -pix_fmt yuv420p P02_109_new.mp4
