import os.path as osp
import numpy as np
from PIL import Image
import cv2
import trimesh
import tqdm
from lib.base_type import ColmapModel
from registration.functions import (
    colmap_image_c2w, colmap_image_w2c,
    umeyama, umeyama_ransac,
)

def visualize_kps(img: np.ndarray, xys, color='red',
                  radius=1, lintType=-1, thickness=2):
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
    }
    img = img.copy()
    for x, y in xys:
        cv2.circle(img, (int(x), int(y)), radius=radius, 
                   color=color_map[color], 
                   lineType=lintType, thickness=thickness)
    return img


def get_points(mix_model, single_model, xyz_only=True, thresh_point=1.0):
    mix_map = build_c2w_map(mix_model)
    single_map = build_c2w_map(single_model)

    pts_mix = []
    pts_single = []
    common = set(mix_map.keys()).intersection(single_map.keys())
    common = sorted(list(common))
    pids1 = set()
    pids2 = set()
    common_images = []
    for key in common:
        if len(mix_map[key].xys) != len(single_map[key].xys):
            continue
        # else:
        #     assert np.abs(mix_map[key].xys - single_map[key].xys).sum() < 1e-3, \
        #         f'diff = {np.abs(mix_map[key].xys - single_map[key].xys).sum()}'
        common_images.append(key)
        num_xys = len(mix_map[key].point3D_ids)
        for i in range(num_xys):
            pid1 = mix_map[key].point3D_ids[i]
            pid2 = single_map[key].point3D_ids[i]
            if pid1 == -1 or pid2 == -1:
                continue 
            if pid1 in pids1 or pid2 in pids2:  # Don't want to count more than once
                # assert pid2 in pids2
                continue
            if np.abs(mix_map[key].xys[i] - single_map[key].xys[i]).sum() > 1e-3:
                continue
            pids1.add(pid1)
            pids2.add(pid2)
            point1 = mix_model.points[pid1]
            point2 = single_model.points[pid2]
            if point1.error > thresh_point or point2.error > thresh_point:
                continue
            if xyz_only:
                pts_mix.append(mix_model.points[pid1].xyz)
                pts_single.append(single_model.points[pid2].xyz)
            else:
                pts_mix.append(mix_model.points[pid1])
                pts_single.append(single_model.points[pid2])
    if xyz_only:
        pts_mix = np.vstack(pts_mix)
        pts_single = np.vstack(pts_single)
    return pts_mix, pts_single, common_images


def get_points_i(ab_map, b_map, mod_ab, mod_b, img_name):
    pts_mix_i = []
    pts_single_i = []
    xys_i = []
    num_xys = len(ab_map[img_name].point3D_ids)
    for i in range(num_xys):
        pid1 = ab_map[img_name].point3D_ids[i]
        pid2 = b_map[img_name].point3D_ids[i]
        if pid1 == -1 or pid2 == -1:
            continue 
        pts_mix_i.append(mod_ab.points[pid1].xyz)
        pts_single_i.append(mod_b.points[pid2].xyz)
        xys_i.append(ab_map[img_name].xys[i])
    pts_mix_i = np.vstack(pts_mix_i)
    pts_single_i = np.vstack(pts_single_i)
    xys_i = np.vstack(xys_i)
    return pts_mix_i, pts_single_i, xys_i


def project_colmap_cam(cam, pts, colmap_img):
    fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
    camMat = np.asarray([
        [fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    pts, _jacob = cv2.projectPoints(pts, rvec=colmap_img.qvec2rotmat(), tvec=colmap_img.tvec, cameraMatrix=camMat,
                           distCoeffs=np.asarray([k1, k2, p1, p2]))
    pts = np.asarray(pts).reshape(-1, 2)
    return pts


def pts_to_scene(pts_ab, pts_b, ab, modelb,
                 s, R, t):
    v_ab_all = np.asarray([v.xyz for v in ab.points.values()])
    v_b_all = np.asarray([v.xyz for v in modelb.points.values()])

    scn = trimesh.scene.Scene()
    for v in (pts_b * s @ R.T + t):
        sp = trimesh.primitives.Sphere(radius=0.05, center=v, subdivisions=1)
        sp.visual.face_colors[:] = [255, 0, 0, 255]  # red for predicted
        scn.add_geometry(sp)

    for v in pts_ab:
        sp = trimesh.primitives.Sphere(radius=0.05, center=v, subdivisions=1)
        sp.visual.face_colors[:] = [0, 255, 0, 255]  # green for common space
        scn.add_geometry(sp)

    v_b_all = v_b_all * s @ R.T + t
    # v_b_select = np.random.choice(v_b_all.shape[0], 5000, replace=False)
    v_b_select = v_b_all # [v_b_select]
    for v in tqdm.tqdm(v_b_select):
        sp = trimesh.primitives.Sphere(radius=0.01, center=v, subdivisions=1)
        sp.visual.face_colors[:] = [125, 0, 255, 255]
        scn.add_geometry(sp)

    v_ab_select = np.random.choice(v_ab_all.shape[0], 5000, replace=False)
    v_ab_select = v_ab_all # [v_ab_select]
    for v in tqdm.tqdm(v_ab_select):
        sp = trimesh.primitives.Sphere(radius=0.01, center=v, subdivisions=1)
        sp.visual.face_colors[:] = [0, 125, 255, 255]
        scn.add_geometry(sp)
    
    return scn


def find_transformation():
    ab = ColmapModel('projects/registration/P02A/model')
    ab.default_vid = 'P02_01'
    modelb = ColmapModel('projects/registration/P02A/P02_02/model/')
    modelb.default_vid = 'P02_02'

    ab_map = build_c2w_map(ab, pose_only=False)
    b_map = build_c2w_map(modelb, pose_only=False)

    pts_ab, pts_b, common_images = get_points(ab, modelb)
    s, R, t, _ = umeyama_ransac(pts_b, pts_ab, k=500, t=0.1, d=0.25)

    print("Generate ply")