import numpy as np
from scipy.spatial.transform import Rotation


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

    if np.abs(np.linalg.det(rot) - 1) < 1e-6:
        raise ValueError("Rotation matrix is not valid")

    pts1_pred = scale * pts2 @ rot.T + transl
    err = np.sum((pts1 - pts1_pred)**2, axis=0)

    return rot, transl, scale, mat, err