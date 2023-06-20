from collections import namedtuple
from typing import List
import os.path as osp
import json
import numpy as np

from scipy.spatial import cKDTree

from libzhifan import io


def common_keypoints(kps1, kps2, thresh=0.8, debug=False):
    """ Find commond 2D keypoints, return their indexes.
    
    Usage:
        kps1 = np.float32([
            [0, 0],
            [1, 1],
            [2, 5]])
        kps2 = np.float32([
            [0, 0.5],
            [2, 5],
            [1, 3],])
        common_keypoints(kps1, kps2)
    
    Args:
        kps1: (N, 2)
        kps2: (M, 2)
        thresh: float, in pixel.
            e.g. the euclidean dist of (0, 0) and (0.5, 0.5) is 0.707
    Returns:
        idx1: (C, )
        idx2: (C, ) 
            index into kps1 and kps2 such that || kps1[idx1[i]] - kps2[idx2[i]] ||^2_2 < thresh
    """
    retVal = namedtuple('retVal', 'idx1 idx2 pts1 pts2 sum_dist avg_dist')
    
    tree = cKDTree(kps1)
    dists, inds = tree.query(kps2, k=1)
    if debug:
        print(f'dists: {dists}')
    keep_inds = dists < thresh
    idx1 = inds[keep_inds]
    idx2 = np.arange(len(kps2))[keep_inds]

    sum_dists = dists[keep_inds].sum()
    avg_dists = sum_dists / len(keep_inds)
    retval = retVal(idx1, idx2, kps1[idx1], kps2[idx2], sum_dists, avg_dists)
    return retval


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
    return True