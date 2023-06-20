import numpy as np
import cv2
import re
from PIL import Image
import tqdm
import trimesh

from colmap_converter.colmap_utils import Camera as ColmapCamera
from colmap_converter.colmap_utils import Image as ColmapImage


""" Visualize """
def draw_points2d(img, xys, *args, **kwargs):
    """
    Args:
        img: np.ndarray, (H, W, 3)
        xys: [N, 2] keypoints
    """
    for x, y in xys:
        img = cv2.circle(
            img, (int(x), int(y)), *args, **kwargs)
    return img

def project_points(pts3d: np.ndarray,
                   w2c: np.ndarray,
                   camera: ColmapCamera,
                   debug=False):
    """ Project 3d points into the image plane, using world-to-camera matrix `w2c`,
    and camera parameters `camera`.

    Args:
        pts3d: (N, 3)
        w2c: (4, 4)
    
    Returns:
        pts2d: (N, 2)
    """
    fx, fy, cx, cy, k1, k2, p1, p2 = camera.params

    pts3d_homo = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    pts3d_cam = (w2c @ pts3d_homo.T).T
    pts3d_cam = pts3d_cam[:, :3]
    if debug:
        print(pts3d_cam[:10, :])
    camera_matrix = np.float32([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    pts2d, jacob = cv2.projectPoints(
        pts3d_cam, np.eye(3), np.zeros(3),
        camera_matrix, distCoeffs=np.float32([k1, k2, p1, p2]))
    return pts2d.squeeze()


def compute_reproj_error(colmap_img: ColmapImage,
                         w2c: np.ndarray,
                         global_points3d: dict,
                         camera: ColmapCamera,
                         visual_debug=False,
                         visual_debug_vid=None) -> float:
    """
    Each colmap_img has pts2d and pts3d,
    we can compute the reprojection error of `w2c` by
    projecting the 3d points into the image plane and compare with the 2d points.

    Args:
        w2c: (4, 4)
        global_points3d: dict of {point3d_id: ColmapPoint3D}
    
    Returns:
        err: error (of this w2c particularly)
    """
    pts3d = []
    gt_pts = []
    for pts3d_id, xy2d in zip(colmap_img.point3D_ids, colmap_img.xys):
        if pts3d_id != -1:
            _pts3d = global_points3d[pts3d_id].xyz
            pts3d.append(_pts3d)
            gt_pts.append(xy2d)
    
    pts3d = np.asarray(pts3d)
    gt_pts = np.asarray(gt_pts)
    proj2d = project_points(pts3d, w2c, camera)
    if visual_debug:
        if visual_debug_vid is None:
            vid = re.search('P\d{2}_\d{2,3}', colmap_img.name)[0]
        else:
            vid = visual_debug_vid
        pid = vid[:3]
        frame_path = re.search('frame_\d{10}.jpg', colmap_img.name)[0]
        image_path = f'/home/skynet/Zhifan/data/epic_rgb_frames/{pid}/{vid}/{frame_path}'
        img = np.asarray(Image.open(image_path))
        print('3d proj:', proj2d[:10, :])
        print('2d gt:', gt_pts[:10, :])
        img = draw_points2d(img, proj2d, radius=1, color=(255, 0, 0), lineType=-1, thickness=2)
        img = draw_points2d(img, gt_pts, radius=1, color=(0,255,0),lineType=-1, thickness=2)
        return img

    diff = gt_pts - proj2d
    w2c_err = (diff ** 2).sum(-1).mean()
    return w2c_err


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