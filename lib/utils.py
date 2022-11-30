from typing import Union, Tuple
import itertools
import re
import glob
import numpy as np
import subprocess
from PIL import Image
from lib.constants import IMG_ROOT


""" File name utils"""

_BASE1 = int(1e13)
_BASE2 = int(1e10)


def convert_to_VFid(path: str) -> str:
    """ Convert any path into 15-digit video-frame-id 
    e.g. /xxx/P01_01/xxx_0111.jpg -> 01 x 10^13 + 01 x 10^10 + 111
    """
    vid = re.search('P\d{2}_\d{2,3}', path)[0]
    fid = int(re.search('\d{10}', path)[0])
    x1, x2 = vid.split('_')
    x1 = int(x1[1:]) * _BASE1
    x2 = int(x2) * _BASE2
    return f'{x1 + x2 + fid:015d}'


def revert_from_VFid(vfid: Union[int, str]) -> Tuple[str, str]:
    """ Given a vf-id generated from `convert_to_VFid`, 
    return a tuple of (vid, frame_10d)
    """
    vfid = int(vfid)
    x1 = f'{vfid // _BASE1:02d}'
    x2 = (vfid % _BASE1) // _BASE2
    if x2 >= 100:
        x2 = f'{x2:03d}'
    else:
        x2 = f'{x2:02d}'
    fid = f'{vfid % _BASE2:010d}'
    pid = f'P{x1}_{x2}'
    return pid, fid


def retrieve_kitchen_frames(pid: str, 
                            visor_img_root=IMG_ROOT,
                            is_epic100=False):
    """ Input pid (e.g. P01), 
        Output all their visor image paths,
            output also their video-frame-id
        
    Args:
        pid: e.g. P01

    Returns:
        - list of source paths
        - list of corresponding video_frame_id
            e.g. P01_02/frame_123.jpg -> vfid = 01x 10^13 + 02x 10^10 + 123
    """
    search = str(visor_img_root/f'{pid}_*')
    parent_list = glob.glob(search)
    vid_len = 7 if is_epic100 else 6  # e.g. P01_101 or P01_03
    parent_list = [
        v for v in parent_list 
        if len(v.split('/')[-1]) == vid_len]
    src_list = itertools.chain.from_iterable(map(lambda x: glob.glob(x+'/*.jpg'), parent_list))
        
    vf_ids = list(map(convert_to_VFid, src_list))
    src_list = [x for _, x in sorted(zip(vf_ids, src_list))]
    vf_ids.sort()
    return src_list, vf_ids
    

""" Image or mask processing """

def visor_to_colmap_mask(in_path: str, 
                         out_path: str,
                         resize=None) -> np.ndarray:
    """
    Converting visor color png mask to binary png mask.

    Args:
        resize: (w, h), original size (854, 480)
    """
    mask = np.asarray(Image.open(in_path))
    h, w = mask.shape
    out = np.ones((h, w, 4), dtype=np.uint8) * 255
    out[mask!=0, :] = [0, 0, 0, 255]
    if out_path is not None:
        out_img = Image.fromarray(out)
        if resize is not None:
            out_img = out_img.resize(resize)
        out_img.save(out_path)
    return out

    
""" FFMPEG related """

def get_video_duration(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)