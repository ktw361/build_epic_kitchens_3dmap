from typing import List, Tuple
import os.path as osp
import json
import re
import numpy as np
from PIL import Image
import cv2

from colmap_converter.colmap_utils import Image as ColmapImage
from colmap_converter.colmap_utils import Camera as ColmapCamera
from lib.base_type import ColmapModel

from libzhifan import io


def colmap_image_w2c(img: ColmapImage) -> np.ndarray:
    raise DeprecationWarning('use lib.common_functions.colmap_image_w2c instead')
    """
    Returns: w2c (4, 4)
    """
    w2c = np.eye(4)
    w2c[:3, :3] = img.qvec2rotmat()
    w2c[:3, 3] = img.tvec
    return w2c


def colmap_image_c2w(img: ColmapImage) -> np.ndarray:
    raise DeprecationWarning('use lib.common_functions.colmap_image_c2w instead')
    """ equiv: np.linalg.inv(colmap_image_w2c(img)) 
    Returns: c2w (4, 4)
    """
    c2w = np.eye(4)
    c2w[:3, :3] = img.qvec2rotmat().T
    c2w[:3, 3] = -img.qvec2rotmat().T @ img.tvec
    return c2w


def colmap_image_loc(img: ColmapImage) -> np.ndarray:
    raise DeprecationWarning('use lib.common_functions.colmap_image_loc instead')
    """
    Returns: camera location (3,) of this image, in world coordinate
    """
    R = img.qvec2rotmat()
    loc = -R.T @ img.tvec
    return loc


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


def umeyama_ransac(src, dst, k, t, d: float, n=3, verbose=False):
    """ Find best dst = c * src * R.T + t

    Args:
        src: (N, 3)
        dst: (N, 3)
        k: int. max number of iterations
        t: float. threshold for the data-consistency criterion
        d: float. ratio of close data values required to assert that a model fits well to data
        n: int. number of points needed for the canonical model
            Ideally 3 points are enough

    Returns:
        c: float. See umeyama()
        R: (3, 3). See umeyama()
        t: (3,). See umeyama()
        all_errs: (N,). The error of each point using the best model
    """
    d = int(d * len(src))
    def pcd_alignment_error(A, B, c, R, t):
        """ mean point-to-point distance between two point clouds, 
        after applying the transformation (c, R, t) to A 

        Returns: (N,)
        """
        diff = c * A @ R.T + t - B
        return np.sqrt((diff ** 2).sum(-1))

    best_model = None
    best_err = np.inf
    for i in range(k):
        inds = np.random.choice(src.shape[0], n, replace=False)
        maybe_model = umeyama(src[inds], dst[inds])
        err = pcd_alignment_error(src, dst, *maybe_model)
        confirmed_inds = np.where(err < t)[0]
        if len(confirmed_inds) >= d:
            better_model = umeyama(src[confirmed_inds], dst[confirmed_inds])
            this_err = pcd_alignment_error(src, dst, *better_model).mean()
            if this_err < best_err:
                best_model = better_model
                best_err = this_err
            if verbose:
                print(f"RANSAC: {len(confirmed_inds)} points are inliers, "
                      f"which is more than {d}, so we accept this model")
        else:
            if verbose:
                print(f"RANSAC: {len(confirmed_inds)} points are inliers, "
                      f"which is less than {d}, so we discard this model,"
                      f"average error is {err.mean()}")
    if best_model is None:
        print("No good model found")
        return None, None, None, None
    all_errs = pcd_alignment_error(src, dst, *best_model)
    c, R, t = best_model
    return c, R, t, all_errs
        

def umeyama(src, dst):
    """
    Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
    with least-squares error.
    Returns (scale factor c, rotation matrix R, translation vector t) such that
      dst = src * c * R.T + t
    if they align perfectly, or such that
      SUM over point i ( | P_i*cR + t - Q_i |^2 )
    is minimised if they don't align perfectly.
    [ref: https://gist.github.com/nh2/bc4e2981b0e213fefd4aaa33edfb3893]
    """
    assert src.shape == dst.shape
    n, dim = src.shape

    centeredP = src - src.mean(axis=0)
    centeredQ = dst - dst.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(src, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = dst.mean(axis=0) - src.mean(axis=0).dot(c*R)

    R = R.T
    return c, R, t


def combine_two_transforms(s1, R1, t1, s2, R2, t2):
    """ dst = s2 * R2 * (s1 * R1 * src + t1) + t2 """
    s = s2 * s1
    R = R2 @ R1
    t = s2 * R2 @ t1 + t2
    return s, R, t


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


""" Utilities """

def get_common_frames(model_a: ColmapModel, 
                      model_b: ColmapModel,
                      frames_a: List[str],
                      frames_b: List[str],
                      return_pos=True) -> Tuple[np.ndarray, np.ndarray]:
    """ Get the common frames between two models.

    Returns:
        images_a, image_b
        if return_pos:
            images_a, image_b: [N, 3]
        else:
            images_a, image_b: list of ColmapImage
    """
    assert len(frames_a) == len(frames_b)
    common_frames = []
    images_a, images_b = [], []
    model_a_dict = {v.name: v for v in model_a.images.values()}
    model_b_dict = {v.name: v for v in model_b.images.values()}
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        if fa in model_a_dict and fb in model_b_dict:
            common_frames.append(fa)
            images_a.append(model_a_dict[fa])
            images_b.append(model_b_dict[fb])
    if return_pos:
        images_a = np.asarray(
            [colmap_image_loc(v) for v in images_a])
        images_b = np.asarray(
            [colmap_image_loc(v) for v in images_b])
    return common_frames, images_a, images_b


def extract_common_images(out_dir, second_model_path: str, second_model_vid: str):
    """
    Args:
        model_path: path to the second model
        model_vid: vid of the second model
    
    Returns:
        common: List[str]
        imgs_dst, imgs_src: np.ndarray
    """
    first_second = ColmapModel(osp.join(out_dir, second_model_vid))  # first + a few second images
    second = ColmapModel(second_model_path)  # pure second 
    frames_first_second = io.read_txt(osp.join(out_dir, second_model_vid, 'image_list.txt'))
    frames_first_second = [v[0] for v in frames_first_second]
    frames_second = [re.search('frame_\d{10}.jpg', v)[0] for v in frames_first_second]
    """ We tranform `second` into `first_second`, hence `second` is src """
    common, imgs_dst, imgs_src = get_common_frames(
        first_second, second, frames_first_second, frames_second, return_pos=True)
    return common, imgs_dst, imgs_src


def write_registration(filename: str, 
                       model_vid: str,
                       s: float, R: np.ndarray, t: np.ndarray):
    has_vid = False
    new_model_info = {
        'model_vid': model_vid,
        'scale': s,
        'rot': R.flatten().tolist(),
        'transl': t.tolist()
    }

    if osp.exists(filename):
        model_infos = io.read_json(filename)
    else:
        model_infos = []

    for model_info in model_infos:
        if model_info['model_vid'] == model_vid:
            model_info.update(new_model_info)
            has_vid = True
            break
    if not has_vid:
        model_infos.append(new_model_info)
    with open(filename, 'w') as fp:
        json.dump(model_infos, fp, indent=2)