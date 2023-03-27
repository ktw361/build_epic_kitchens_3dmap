from typing import List, Tuple
from colmap_converter.colmap_utils import Image
import numpy as np
import pyquaternion as pyq
from scipy.spatial.transform import Rotation


def compute_relative_pose(imgs: List[Image]) -> Tuple[List[pyq.Quaternion], np.ndarray]:
    """Compute the relative pose w.r.t the first image.
    Args:
        imgs: list of N+1 images
    Return:
        q_rel: list of N relative quaternions
        t_rel: [N-1, 3]
    """
    img0 = imgs[0]
    q0 = pyq.Quaternion(img0.qvec)
    q0_inv = q0.inverse
    t0 = img0.tvec
    q_rel = []
    t_rel = []
    for img in imgs[1:]:
        q = pyq.Quaternion(img.qvec)
        t = img.tvec
        q_rel.append(q * q0_inv)
        t_rel.append(t - t0)
    t_rel = np.array(t_rel)
    return q_rel, t_rel


def compute_pose_errors(q_rel1: List[pyq.Quaternion], t_rel1: np.ndarray, 
                        q_rel2: List[pyq.Quaternion], t_rel2: np.ndarray
                        ) -> Tuple[List[pyq.Quaternion], np.ndarray]:
    """ 
    Args:
        q_rel1: list of N quaternions
        t_rel1: [N, 3], typically the ground-truth poses
        q_rel2: list of N quaternions
        t_rel2: [N, 3]
    
    Returns:
        quat_error: list of N quaternions
        transl_errors: [N,]
        transl_errors_ratio: [N,], transl_errors / t_rel1
    """
    quat_error = []
    for q1, q2 in zip(q_rel1, q_rel2):
        quat_error.append(q1 * q2.inverse)
    transl_errors = t_rel1 - t_rel2
    transl_errors = np.linalg.norm(transl_errors, axis=1)
    t_rel2_norm = np.linalg.norm(t_rel2, axis=1)
    transl_errors_ratio = transl_errors / t_rel2_norm
    return quat_error, transl_errors, transl_errors_ratio


def solve_scale(t_rel1: np.ndarray, t_rel2: np.ndarray):
    """ 
    Return: solution to minimize (X - s Y)^2, and the final error 
        transl_errors: [N,]
        transl_errors_ratio: [N,], transl_errors / t_rel1
        scale: float
    """
    YtY = np.dot(t_rel2.T, t_rel2)
    XtY = np.dot(t_rel1.T, t_rel2)
    YtX = np.dot(t_rel2.T, t_rel1)
    s = np.trace(XtY + YtX) / (2 * np.trace(YtY))
    error = np.linalg.norm(t_rel1 - s * t_rel2, axis=1)
    error_ratio = error / np.linalg.norm(t_rel1, axis=1)
    return error, error_ratio, s


def compute_minimal_pose_errors_with_scale(q_rel1: List[pyq.Quaternion], t_rel1: np.ndarray,
                                           q_rel2: List[pyq.Quaternion], t_rel2: np.ndarray
                                           ) -> Tuple[List[pyq.Quaternion], np.ndarray]:
    """ If we can scale model_2 to match model_1, then we can find the scale
    s.t. sum of squared distance is minimal.
    The solution to minimize (X - s Y)^2 is 
        Tr(Y^t X + X^t Y) 
    s = -----------------
         2 * Tr(Y^t Y)

    Args:
        q_rel1: list of N quaternions
        t_rel1: [N, 3], typically the ground-truth poses
        q_rel2: list of N quaternions
        t_rel2: [N, 3]
    
    Returns:
        quat_error: list of N quaternions
        transl_errors: [N,]
        transl_errors_ratio: [N,], transl_errors / t_rel1
        scale: float
    """
    quat_error = []
    for q1, q2 in zip(q_rel1, q_rel2):
        quat_error.append(q1 * q2.inverse)
    transl_errors, transl_errors_ratio, scale = solve_scale(t_rel1, t_rel2)
    return quat_error, transl_errors, transl_errors_ratio, scale


def model_compute_relative_poses(model_1, model_2, common_input):
    """
    Args:
        model_1 / model_2 : SparseProj 
        common_input: iterable of image names
    """
    common_set = set(common_input)
    reg1 = set(v for v in model_1.images_registered_names if v in common_set)
    reg2 = set(v for v in model_2.images_registered_names if v in common_set)
    overlap = reg1.intersection(reg2)
    imgs1 = [model_1.images_registered_names[v] for v in overlap]
    imgs2 = [model_2.images_registered_names[v] for v in overlap]
    rel_poses1 = compute_relative_pose(imgs1)
    rel_poses2 = compute_relative_pose(imgs2)
    return *rel_poses1, *rel_poses2


def model_compute_pose_errors(model_1, model_2, common_input, solve_scale: bool) -> List[Tuple[pyq.Quaternion, float]]:
    common_set = set(common_input)
    reg1 = set(v for v in model_1.images_registered_names if v in common_set)
    reg2 = set(v for v in model_2.images_registered_names if v in common_set)
    overlap = reg1.intersection(reg2)
    imgs1 = [model_1.images_registered_names[v] for v in overlap]
    imgs2 = [model_2.images_registered_names[v] for v in overlap]
    rel_poses1 = compute_relative_pose(imgs1)
    rel_poses2 = compute_relative_pose(imgs2)
    if solve_scale:
        quat_error, transl_errors, transl_errors_ratio, scale = compute_minimal_pose_errors_with_scale(
            *rel_poses1, *rel_poses2)
        return overlap, quat_error, transl_errors, transl_errors_ratio, scale
    else:
        quat_error, transl_errors, transl_errors_ratio = compute_pose_errors(
            *rel_poses1, *rel_poses2)
        return overlap, quat_error, transl_errors, transl_errors_ratio, None



def compute_sim3_transform(pts1: np.ndarray,
                           pts2: np.ndarray) -> np.ndarray:
    """
    Computes SIM(3) transformation 
        pts1 = s * R * pts2 + t

    Args:
        pts1 / pts2: (N, 3) points already have correspondence.
    
    Returns: convention is "column-vector to the right"
        R: (3, 3)
        t: (1, 3)
        s: float
        mat: (4, 4), composed <R, t, s>
        err: (N,) squared distance
    """
    c1 = pts1.mean(0)
    c2 = pts2.mean(0)
    pts1_centered = pts1 - c1
    pts2_centered = pts2 - c2
    avg_norm1 = np.linalg.norm(pts1_centered, axis=0).mean()
    avg_norm2 = np.linalg.norm(pts2_centered, axis=0).mean()
    scale = avg_norm1 / avg_norm2
    pts2_scaled = pts2_centered * scale
    transl = (c1 - c2) / scale
    _rot, _ = Rotation.align_vectors(pts1_centered, pts2_scaled)
    rot = _rot.as_matrix()
    mat = np.empty((4, 4), dtype=pts1.dtype)
    mat[:3, :3] = rot
    mat[:3, -1] = transl
    mat[-1, -1] = scale

    if np.abs(np.linalg.det(rot) - 1) > 1e-6:
        raise ValueError(
            "Rotation matrix is not valid, det(R) = %f" % np.linalg.det(rot))

    pts1_pred = scale * pts2 @ rot.T + transl
    err = np.sum((pts1 - pts1_pred)**2, axis=0)

    return rot, transl, scale, mat, err