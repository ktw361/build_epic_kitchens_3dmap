import torch
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion
import numpy as np

def qtvec2mat(qvec: np.ndarray, tvec: np.ndarray) -> torch.Tensor:
    n = len(qvec)
    R = quaternion_to_matrix(torch.from_numpy(qvec))  # qvec2rotmat(qvec)
    mat = torch.eye(4).view(-1, 4, 4).tile([n, 1, 1])
    mat[:, :3, :3] = R
    mat[:, :3, 3] = torch.from_numpy(tvec)
    return mat.float()


def mat2qtvec(mat: torch.Tensor) -> np.ndarray:
    qvec = matrix_to_quaternion(mat[:, :3, :3]).numpy()
    tvec = mat[:, :3, 3].numpy()
    qtvecs = np.concatenate([qvec, tvec], 1)
    return qtvecs


def get_inverse_transform(s, R, t) -> torch.Tensor:
    """ 
    input matrix of {RsP + st} is [R, t, 1/s],
    and it's inverse is [R.T, -R.T @ s*t, s]
    """
    mat = np.eye(4)
    mat[:3, :3] = R.T
    mat[:3, 3] = - R.T @ (s*t)
    mat[-1, -1] = s
    return torch.from_numpy(mat).float()


def transform_images(image_arr, scale, rot, transl):
    """
    Args:
        image_arr: (N, 7) of (qw, qx, qy, qz, x, y, z)
    
    Returns:
        (N, 7) of (qw, qx, qy, qz, x, y, z)
    """
    w2c = qtvec2mat(image_arr[:, :4], image_arr[:, 4:])
    inv_transf = get_inverse_transform(s=scale, R=rot, t=transl/scale)  # {RsP+t} == {RsP + s*(t/s)}
    new_w2c = w2c @ inv_transf
    new_images = mat2qtvec(new_w2c)  # (N, 7)
    return new_images