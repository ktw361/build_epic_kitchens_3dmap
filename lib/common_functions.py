import re
import numpy as np
from colmap_converter.colmap_utils import Image as ColmapImage
from lib.base_type import ColmapModel


def colmap_image_w2c(img: ColmapImage) -> np.ndarray:
    """
    Returns: w2c (4, 4)
    """
    w2c = np.eye(4)
    w2c[:3, :3] = img.qvec2rotmat()
    w2c[:3, 3] = img.tvec
    return w2c


def colmap_image_c2w(img: ColmapImage) -> np.ndarray:
    """ equiv: np.linalg.inv(colmap_image_w2c(img)) 
    Returns: c2w (4, 4)
    """
    c2w = np.eye(4)
    c2w[:3, :3] = img.qvec2rotmat().T
    c2w[:3, 3] = -img.qvec2rotmat().T @ img.tvec
    return c2w


def colmap_image_loc(img: ColmapImage) -> np.ndarray:
    """
    Returns: camera location (3,) of this image, in world coordinate
    """
    R = img.qvec2rotmat()
    loc = -R.T @ img.tvec
    return loc


def build_c2w_map_str(model: ColmapModel, pose_only=False):
    """ input model is enhance with `default_vid`

    Args:
        key_type: 'name' or 'frame'

    key: P01_01/frame_.jpg
        value: (4, 4) ndarray, or ColmapImage
    """
    mp = dict()
    default_vid = model.default_vid
    for img in model.ordered_images:
        if not img.name.startswith('P'):
            name = f'{default_vid}/{img.name}'
        else:
            name = img.name
        if pose_only:
            mp[name] = colmap_image_c2w(img)
        else:
            mp[name] = img
    return mp


def build_c2w_map_int(model, pose_only=False):
    """ This is for single video only

    key: frame number in int-type
    value: (4, 4) ndarray, or ColmapImage
    """
    mp = dict()
    for img in model.ordered_images:
        frame = int(re.search('\d{10,}', img.name)[0])
        if pose_only:
            mp[frame] = colmap_image_c2w(img)
        else:
            mp[frame] = img
    mp = dict(sorted([(k, v) for k, v in mp.items()], key=lambda kv: kv[0]))
    return mp