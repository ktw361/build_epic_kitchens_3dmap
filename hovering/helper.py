import os
import numpy as np
from PIL import Image
import open3d as o3d
from open3d.visualization import rendering

from lib.base_type import ColmapModel

from hovering.o3d_line_mesh import LineMesh


class Helper:
    base_colors = {
        'white': [1, 1, 1, 1],
        'red': [1, 0, 0, 1],
    }

    def __init__(self, 
                 point_size):
        self.point_size = point_size
    
    def material(self, color: str):
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = self.base_colors[color]
        material.point_size = self.point_size
        return material


def get_o3d_pcd(model: ColmapModel):
    pcd_np = np.asarray([v.xyz for v in model.points.values()])
    pcd_rgb = np.asarray([v.rgb / 255 for v in model.points.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    return pcd



def get_frustum(sz=1.0, line_radius=0.15,
                colmap_image=None, colmap_camera=None):
    """
    """
    cen = [0, 0, 0]
    wid = sz
    if colmap_camera is not None:
        hei = wid * colmap_camera.height / colmap_camera.width
    else:
        hei = wid
    tl = [wid, hei, sz]
    tr = [-wid, hei, sz]
    br = [-wid, -hei, sz]
    bl = [wid, -hei, sz]
    points = np.float32([cen, tl, tr, br, bl])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],]
    line_mesh = LineMesh(
        points, lines, colors=[1, 0, 0], radius=line_radius)
    line_mesh.merge_cylinder_segments()
    frustum = line_mesh.cylinder_segments[0]

    if colmap_image is not None:
        w2c = np.eye(4)
        w2c[:3, :3] = colmap_image.qvec2rotmat()
        w2c[:3, -1] = colmap_image.tvec
        c2w = np.linalg.inv(w2c)

        frustum = frustum.transform(c2w)
    return frustum


def get_cam_pos(colmap_image):
    cen = np.float32([0, 0, 0, 1])
    w2c = np.eye(4)
    w2c[:3, :3] = colmap_image.qvec2rotmat()
    w2c[:3, -1] = colmap_image.tvec
    c2w = np.linalg.inv(w2c)
    pos = c2w @ cen
    return pos[:3]


def get_trajectory(pos_history, num_line=6, line_radius=0.15,):
    """ pos_history: absolute position history
    """
    pos_history = np.asarray(pos_history)[-num_line:]
    colors = [
        [0, 0, 1],
        [0.2, 0, 0.8],
        [0.4, 0, 0.6],
        [0.6, 0, 0.4],
        [0.8, 0, 0.2],
        [1, 0, 0],
    ]
    colors = colors[-len(pos_history):]
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    line_mesh.merge_cylinder_segments()
    path = line_mesh.cylinder_segments[0]
    return path


def read_original(colmap_img, frame_root):
    return np.asarray(Image.open(os.path.join(frame_root, colmap_img.name)))
