from typing import List
import os
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
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
        'blue': [0, 0, 1,1],
        'green': [0, 1, 0,1],
        'yellow': [1, 1, 0,1],
        'purple': [0.2, 0.2, 0.8, 1]
    }

    def __init__(self, point_size):
        self.point_size = point_size
    
    def material(self, color: str, shader="defaultUnlit") -> rendering.MaterialRecord:
        """
        Args:
            shader: e.g.'defaultUnlit', 'defaultLit', 'depth', 'normal'
                see Open3D: cpp/open3d/visualization/rendering/filament/FilamentScene.cpp#L1109
        """
        material = rendering.MaterialRecord()
        material.shader = shader
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
                camera_width=None,
                frustum_color=[1, 0, 0]) -> o3d.geometry.TriangleMesh:
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
        points, lines, colors=frustum_color, radius=line_radius)
    line_mesh.merge_cylinder_segments()
    frustum = line_mesh.cylinder_segments[0]

    if colmap_image is not None:
        w2c = np.eye(4)
        w2c[:3, :3] = colmap_image.qvec2rotmat()
        w2c[:3, -1] = colmap_image.tvec
        c2w = np.linalg.inv(w2c)
        frustum = frustum.transform(c2w)
    return frustum


def get_frustum_green(*args, **kwargs):
    kwargs['frustum_color'] = [0, 1, 0]
    return get_frustum(*args, **kwargs)


def get_frustum_fixed(*args, **kwargs):
    kwargs['frustum_color'] = [0, 0, 1]
    _ = kwargs.pop('colmap_image')
    frustum = get_frustum(*args, **kwargs)
    w2c = np.eye(4)
    w2c[0:3,:] = [[ 0.98857542  ,0.13693023,  0.06299805,  1.0372356 ],
                [-0.13046984  ,0.98667212, -0.09724064,  2.5839325 ],
                [-0.07547361  ,0.08791036,  0.99326507, -0.6469172 ]]
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
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    line_mesh.merge_cylinder_segments()
    path = line_mesh.cylinder_segments[0]
    return path


def get_pretty_trajectory(pos_history,
                          num_line=6,
                          line_radius=0.15,
                          darkness=1.0,
                          ) -> List[o3d.geometry.TriangleMesh]:
    """ pos_history: absolute position history
    """
    def generate_jet_colors(n, darkness=0.6):
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0, vmax=n-1)
        colors = cmap(norm(np.arange(n)))
        # Convert RGBA to RGB
        colors_rgb = []
        for color in colors:
            colors_rgb.append(color[:3] * darkness)

        return colors_rgb

    pos_history = np.asarray(pos_history)[-num_line:]
    colors = generate_jet_colors(len(pos_history), darkness)
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    return line_mesh.cylinder_segments


def read_original(colmap_img: BaseImage, frame_root: str) -> np.ndarray:
    """ Read epic-kitchens original image from frame_root
    """
    return np.asarray(Image.open(os.path.join(frame_root, colmap_img.name)))


""" Obtain Viewpoint from Open3D GUI """
def parse_o3d_gui_view_status(status: dict, render: rendering.OffscreenRenderer):
    """ Parse open3d GUI's view status and convert to OffscreenRenderer format.
    This will do the normalisation of front and compute eye vector (updated version of front)

    
    Args:
        status: Ctrl-C output from Open3D GUI
        render: OffscreenRenderer
    Output:
       params for render.setup_camera(fov, lookat, eye, up) 
    """
    cam_info = status['trajectory'][0]
    fov = cam_info['field_of_view']
    lookat = np.asarray(cam_info['lookat'])
    front = np.asarray(cam_info['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(cam_info['up'])
    zoom = cam_info['zoom']
    """ 
    See Open3D/cpp/open3d/visualization/visualizer/ViewControl.cpp#L243: 
        void ViewControl::SetProjectionParameters()
    """
    right = np.cross(up, front) / np.linalg.norm(np.cross(up, front))
    view_ratio = zoom * render.scene.bounding_box.get_max_extent()
    distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
    eye = lookat + front * distance
    return fov, lookat, eye, up


def set_offscreen_as_gui(render: rendering.OffscreenRenderer, status: dict):
    """ Set offscreen renderer as GUI's view status
    """
    fov, lookat, eye, up = parse_o3d_gui_view_status(status, render)
    render.setup_camera(fov, lookat, eye, up)