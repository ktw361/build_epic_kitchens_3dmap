from collections import namedtuple
import os.path as osp
import numpy as np
from lib.base_type import ColmapModel
from lib.common_functions import build_c2w_map_str

from registration.functions import common_keypoints


class RegistrateManager:
    
    def __init__(self, ref_vid: str, cur_vid: str, run_dir):
        """
            e.g. run_dir='./projects/registration/P04A_01/'
        """
        self.ref_vid = ref_vid
        self.cur_vid = cur_vid
        self.run_dir = run_dir

        self.ref = ColmapModel(osp.join(run_dir, cur_vid))
        self.cur = self.get_single_model(cur_vid)
        self.ref_map = build_c2w_map_str(self.ref, ref_vid)
        self.cur_map = build_c2w_map_str(self.cur, cur_vid)

    @staticmethod
    def get_single_model(vid,
                         root_dir='/home/skynet/Zhifan/epic_fields_full/skeletons',
                         suffix='sparse/0'):
        model_path = osp.join(root_dir, f'{vid}_low', suffix)
        return ColmapModel(model_path)


def common_points3d(manager: RegistrateManager, 
                    point_error_thresh=1.0):
    """
    Returns:
        xyzs_ref: (N, 3)
        xyzs_cur: (N, 3)
    """
    retVal = namedtuple('retVal', 'xyz_ref xyz_cur points_ref points_cur common_images')
    ref, cur = manager.ref, manager.cur
    ref_map = manager.ref_map
    cur_map = manager.cur_map

    xyz_ref, xyz_cur = [], []
    pts_ref, pts_cur = [], []

    common_keys = set(ref_map.keys()).intersection(cur_map.keys())
    common_keys = sorted(list(common_keys))
    pids1, pids2 = set(), set()
    common_images = []

    for key in common_keys:
        common_images.append(key)
        _ret = common_keypoints(ref_map[key].xys, cur_map[key].xys)
        ref_idx, cur_idx = _ret.idx1, _ret.idx2
        for ref_i, cur_i in zip(ref_idx, cur_idx):
            pid1 = ref_map[key].point3D_ids[ref_i]
            pid2 = cur_map[key].point3D_ids[cur_i]
            if pid1 == -1 or pid2 == -1:
                continue
            if pid1 in pids1 or pid2 in pids2:  # Don't want to count more than once
                # assert pid2 in pids2
                continue
            pids1.add(pid1); pids2.add(pid2)
            point1 = ref.points[pid1]
            point2 = cur.points[pid2]
            if point1.error > point_error_thresh or point2.error > point_error_thresh:
                continue
            xyz_ref.append(point1.xyz)
            xyz_cur.append(point2.xyz)
            pts_ref.append(point1)
            pts_cur.append(point2)
    
    xyz_ref = np.vstack(xyz_ref) if len(xyz_ref) > 0 else np.zeros((0, 3))
    xyz_cur = np.vstack(xyz_cur) if len(xyz_cur) > 0 else np.zeros((0, 3))
    retval = retVal(xyz_ref, xyz_cur, pts_ref, pts_cur, common_images)
    return retval
