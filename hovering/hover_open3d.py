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

from libzhifan import epylab


EPIC_WIDTH = 456
EPIC_HEIGHT = 256


class HoverRunner:
    def __init__(self):
        render = rendering.OffscreenRenderer(1920, 1080)


class RunP34_104(HoverRunner):

    frames_root = '/home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104'
    extrinsic = np.float32([
        [-1, 0, 0, -10],
        [0, -1, 0, -8],
        [0, 0, 1, 18],
        [0, 0, 0, 1]
    ])
    fmt = './P34_104_sparse/%010d.png'  # Save format


    def __init__(self, is_enhance=True):
        if is_enhance:
            self.model = ColmapModel('/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P34_104/enhanced_model//')
        else:
            self.model = ColmapModel('/home/skynet/Zhifan/build_kitchens_3dmap/projects/ahmad/P34_104/sparse_model/')
        self.is_enhance = is_enhance
    
    def test_frame(self, is_enhance=True) -> np.ndarray:
        render.scene.clear_geometry()

        if self.is_enhance:
            psize = 0.5 # enhance
        else:
            psize = 3 # sparse
        helper = Helper(point_size=psize)

        white = helper.material('white')
        red = helper.material('red')

        render.scene.add_geometry('pcd', pcd, white)
        # render.scene.add_geometry('frustum', line_set, red)
        render.scene.set_background([1, 1, 1, 2])

        # cam = o3d.camera.PinholeCameraIntrinsic(
        #     width=1920, height=-1, fx=1, fy=1, cx=0, cy=0)
        cam = o3d.camera.PinholeCameraIntrinsic()
        render.setup_camera(cam, self.extrinsic)

        # render.setup_camera(60.0, [0, 0, 0], [D/2, D, 2*D], [0, 0.2, 1])
        # render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
        #                                  75000)
        # render.scene.scene.enable_sun_light(True)
        render.scene.show_axes(False)

        img = render.render_to_image()
        img = np.asarray(img)

        img_ids = self.model.ordered_image_ids
        img = model.get_image_by_id(1)
        line_set = get_frustum(
            sz=1, line_radius=0.02, colmap_image=img, colmap_camera=model.camera)
        ii = read_original(img, frame_root='/home/skynet/Zhifan/data/epic_rgb_frames/P34/P34_104')
        img[-EPIC_HEIGHT:, 500:500+EPIC_WIDTH] = ii

        return img

    def run_all():
        colmap_camera = self.model.camera

        traj_len = 5
        pos_history = []
        for img_id in tqdm(model.ordered_image_ids[:]):
            
            c_img = model.get_image_by_id(img_id)
            frame_idx = int(re.search('\d{10}', c_img.name)[0])
            line_set = get_frustum(
                sz=1, line_radius=0.02, colmap_image=c_img, colmap_camera=colmap_camera)
            pos_history.append(get_cam_pos(c_img))
            
            if len(pos_history) > 2:
                traj = get_trajectory(pos_history, num_line=traj_len, line_radius=0.02)
                render.scene.add_geometry('traj', traj, red)    
            render.scene.add_geometry('frustum', line_set, red)
            img = render.render_to_image()
            img = np.asarray(img)
            
            ii = read_original(c_img, frame_root=frame_root)
            img[-epic_h:, 500:500+epic_w] = ii
            
            Image.fromarray(img).save(fmt % frame_idx)
            # o3d.io.write_image(fmt % frame_idx, img, 9)

            if len(pos_history) > 2:
                render.scene.remove_geometry('traj')
            render.scene.remove_geometry('frustum')

        # Gen output
        seq = sorted(glob.glob('P34_104_sparse/*.png'))
        clip = editor.ImageSequenceClip(seq, fps=12)
        clip.write_videofile('P34_104_sparse.mp4')