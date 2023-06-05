from typing import List
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from colmap_converter.colmap_utils import BaseImage
from lib.base_type import ColmapModel
from hovering.o3d_line_mesh import LineMesh
from registration.point_registrate import build_c2w_map
from registration.functions import colmap_image_c2w


def generate_colormap(num_colors):
    colormap = plt.get_cmap("hsv", num_colors)
    color_dict = {}
    inds = np.arange(num_colors)
    np.random.seed(42)
    inds = np.random.permutation(inds)
    for i in range(num_colors):
        color_name = f"color_{i}"
        color_dict[color_name] = colormap(inds[i])[:3]
    return color_dict

class Quantizer:
    def __init__(self, x_len=1.0, y_len=1.0, z_len=1.0):
        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len
    def to_ijk(self, xyz: np.ndarray) -> tuple:
        return tuple(np.int32(xyz / np.float32([self.x_len, self.y_len, self.z_len])).flatten().tolist())
    
    def to_xyz(self, ijk: np.ndarray) -> np.ndarray:
        return np.float32(ijk) * np.float32([self.x_len, self.y_len, self.z_len])

def act_location(video_actions: pd.DataFrame, dense_model: ColmapModel, vid,
                lookAt_distance=5.0,
                 to_xyz=True, with_time=False):
    act2cimg = dict()
    dense_model.default_vid = vid
    frame2cimg = build_c2w_map(dense_model)
    for i, row in video_actions.iterrows():
        start_frame = row.start_frame
        act = f'{row.verb} {row.noun}'
        cimg = None
        for f in range(row.start_frame, row.stop_frame+1):
            key = f'{vid}/frame_{start_frame:010d}.jpg'
            if key not in frame2cimg:
                continue
            cimg = frame2cimg[key]
            break
        if cimg is None:
            continue
        if act not in act2cimg:
            act2cimg[act] = []
        if to_xyz:
            lookAt_h = np.float32([0, 0, lookAt_distance, 1.0])
            lookAt_world = colmap_image_c2w(cimg) @ lookAt_h
            lookAt_world = lookAt_world[:3] / lookAt_world[3]
            if with_time:
                act2cimg[act].append((lookAt_world, row.start_timestamp))
            else:
                act2cimg[act].append(lookAt_world)
        else:
            act2cimg[act].append(cimg)
    return act2cimg

def act_hist3d(video_actions: pd.DataFrame, dense_model: ColmapModel, vid, lookAt_distance=5.0,
               quantizer: Quantizer = None, act2xyz=None):
    if quantizer is None:
        quantizer = Quantizer()
    if act2xyz is None:
        act2xyz = act_location(video_actions, dense_model, vid, lookAt_distance=lookAt_distance, to_xyz=True)
    hist3d = dict()
    for act, xyzs in act2xyz.items():
        for xyz in xyzs:
            ijk = quantizer.to_ijk(xyz)
            if ijk not in hist3d:
                hist3d[ijk] = []
            hist3d[ijk].append(act)
    
    hist3d = {k: v for k, v in sorted(hist3d.items(), key=lambda item: -len(item[1]))}  # sort by number of actions
    return hist3d


def get_bubble(loc, sz=1.0) -> o3d.geometry.TriangleMesh:
    bubble = o3d.geometry.TriangleMesh.create_sphere(sz, 8)
    bubble = bubble.translate(np.float32(loc))
    return bubble


def get_n_bubbles(loc, n, sz=1.0, gap=2.0) -> List[o3d.geometry.TriangleMesh]:
    bubbles = []
    for i in range(n):
        bubble = o3d.geometry.TriangleMesh.create_sphere(sz, 8)
        bubble = bubble.translate(np.float32([i*gap, 0, 0]))
        bubble = bubble.translate(np.float32(loc))
        bubbles.append(bubble)
    return bubbles


def build_act_infos(hist3d, quantizer: Quantizer, topk=16,
                    bubble_sz=0.25, bubble_gap=0.5):
    """
    (bubble, act, color: 4)
    """
    color_dict = generate_colormap(topk)
    color_codes = [(*v, 1.0) for v in color_dict.values()]
    cnt = 0
    # triplets = []
    infos = []
    for ijk, acts in hist3d.items():
        n = len(acts)
        bubbles = get_n_bubbles(quantizer.to_xyz(ijk), n, sz=bubble_sz, gap=bubble_gap)
        for i, act in enumerate(acts):
            if cnt >= topk:
                break
            infos.append((bubbles[i], act, color_codes[cnt]))
            cnt += 1
        if cnt >= topk:
            break
    return infos


def get_pointer(sz=1.0, 
                line_radius=0.15,
                colmap_image: BaseImage = None) -> o3d.geometry.TriangleMesh:
    """
    Args:
        sz: float, size (width) of the frustum
        colmap_image: ColmapImage, if not None, the frustum will be transformed
            otherwise the frustum will "lookAt" +z direction
    """
    cen = [0, 0, sz*1]
    tgt = [0, 0, sz*5]
    points = np.float32([cen, tgt])
    lines = [[0, 1],]
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


def text_3d(text, 
            pos, 
            direction=None, 
            degree=0.0, 
            color=(255, 255, 255),
            font_size=16, 
            font='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=color)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    # pcd.transform(trans)
    return pcd
