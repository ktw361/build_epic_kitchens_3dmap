import os
import numpy as np
from PIL import Image
import open3d as o3d
from open3d.visualization import rendering

from colmap_converter.colmap_utils import (
    BaseImage, Point3D, Camera
)
from lib.base_type import ColmapModel

from hovering.o3d_line_mesh import LineMesh


class Helper:
    base_colors = {
        'white': [1, 1, 1, 0.8],
        'red': [1, 0, 0, 1],
    }

    def __init__(self, 
                 point_size):
        self.point_size = point_size
    
    def material(self, color: str) -> rendering.MaterialRecord:
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = self.base_colors[color]
        material.point_size = self.point_size
        return material


def get_o3d_pcd(model: ColmapModel) -> o3d.geometry.PointCloud:
    pcd_np = np.asarray([v.xyz for v in model.points.values()])
    pcd_rgb = np.asarray([v.rgb / 255 for v in model.points.values()])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
    return pcd


def get_frustum(sz=1.0, 
                line_radius=0.15,
                colmap_image: BaseImage = None, 
                camera_height=None,
                camera_width=None) -> o3d.geometry.TriangleMesh:
    """
    Args:
        sz: float, size (width) of the frustum
        colmap_image: ColmapImage, if not None, the frustum will be transformed
            otherwise the frustum will "lookAt" +z direction
    """
    cen = [0, 0, 0]
    wid = sz
    if camera_height is not None and camera_width is not None:
        hei = wid * camera_height / camera_width
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


def get_cam_pos(colmap_image: BaseImage) -> np.ndarray:
    """ Get camera position in world coordinate system
    """
    cen = np.float32([0, 0, 0, 1])
    w2c = np.eye(4)
    w2c[:3, :3] = colmap_image.qvec2rotmat()
    w2c[:3, -1] = colmap_image.tvec
    c2w = np.linalg.inv(w2c)
    pos = c2w @ cen
    return pos[:3]


def get_trajectory(pos_history,
                   num_line=6,
                   line_radius=0.15
                   ) -> o3d.geometry.TriangleMesh:
    """ pos_history: absolute position history
    """
    pos_history = np.asarray(pos_history)[-num_line:]
    colors = [0, 0, 0.6]
    # colors = colors[-len(pos_history):]
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    line_mesh.merge_cylinder_segments()
    path = line_mesh.cylinder_segments[0]
    return path


def read_original(colmap_img: BaseImage, frame_root: str) -> np.ndarray:
    """ Read epic-kitchens original image from frame_root
    """
    return np.asarray(Image.open(os.path.join(frame_root, colmap_img.name)))


import roma
import torch

def colmap_quaternion_slerp(q0, q1, steps: list) -> np.ndarray:
    """
    # Verify the rotation matrix are the same
    #   p02.model.get_image_by_id(p02.model.ordered_image_ids[0]).qvec2rotmat()
    #   roma.unitquat_to_rotmat(q0)
    
    Note: colmap quaternion is (w, x, y, z),
    internally we convert to roma quaternion whic his (x, y, z, w)
    
    Example Usage:
    ```
    q0 = p02.model.get_image_by_id(p02.model.ordered_image_ids[0]).qvec
    q1 = p02.model.get_image_by_id(p02.model.ordered_image_ids[1]).qvec    
    steps = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8])
    qs = colmap_quaternion_slerp(q0, q1, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ```
    
    Args:
        q0/q1: colmap quaternion (w, x, y, z)
        steps: list of float.
    Returns:
        qs: [N, 4]
    """
    q0 = torch.as_tensor(q0).view(-1, 4)
    q1 = torch.as_tensor(q1).view(-1, 4)
    steps = torch.as_tensor(steps)
    
    # WXZY -> XYZW
    q0 = q0[:, [1, 2, 3, 0]]
    q1 = q1[:, [1, 2, 3, 0]]
    # Or faster: https://github.com/naver/roma/blob/ace63568adb09102984674abbe52e9ba6d562702/roma/utils.py#L316
    interp = roma.unitquat_slerp(q0, q1, steps, shortest_path=True)  # torch.Tensor, [shape len(steps), 4]
    interp = interp.view(len(steps), 4)[:, [3, 0, 1, 2]].numpy()
    return interp


# def compute_scene_rotation(o3d_view_dict: dict) -> np.ndarray:
#     """ Open a open3d gui, rotation the scene s.t. it aligns the xyz-axis,
#     press Ctrl-C, paste the output dict to this function.

#     Assuming bydefault camera is 
#         front=[0, 1, 0],
#         lookat=[0, 0, 0],
#         up=[0, 0, 1]

#     By calculating how the camera rotates, we know how the scene rotates ;)
#     And rotating the scene is the inverse of rotating the camera.

#     Args:
#         o3d_view_dict: 
#             e.g. {
#                 "boundingbox_max" : [ 48.408693742688349, 23.676346163790416, 69.1777191121014 ],
#                 "boundingbox_min" : [ -19.596865703896899, -8.9897621246759112, -6.6309501390409018 ],
#                 "field_of_view" : 60.0,
#                 "front" : [ -0.94736648010521285, 0.24989216312824389, -0.20012660787648026 ],
#                 "lookat" : [ 0.0, 0.0, 0.0 ],
#                 "up" : [ 0.11338819398419565, -0.32268615457893568, -0.93968971640007948 ],
#                 "zoom" : 0.13800000000000001
#             }
#     """
#     normed = lambda x: x / np.linalg.norm(x)
#     def get_camera_rotation_matrix(front, lookat, up):
#         lookat = np.asarray(lookat)
#         up = np.asarray(up)
#         front = np.asarray(front)
#         Ra = normed(lookat - front)
#         Rb = normed(up)
#         Rc = np.cross(Ra, Rb)
#         return np.stack([Rc, Rb, Ra], axis=1)
#     front = np.asarray(o3d_view_dict['front'])
#     up = np.asarray(o3d_view_dict['up'])
#     lookat = np.asarray(o3d_view_dict['lookat'])
#     assert np.abs(lookat).sum() < 1e-6, "lookat should be [0, 0, 0]"
#     R0 = get_camera_rotation_matrix(
#         [0, 1, 0], [0, 0, 0], [0, 0, 1])
#     R1 = get_camera_rotation_matrix(
#         front, lookat, up)
#     return R1.T @ R0
    
