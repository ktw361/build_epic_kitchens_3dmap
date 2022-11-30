import os
import argparse
import os.path as osp
from lib.utils import retrieve_kitchen_frames, revert_from_VFid
from lib.constants import IMG_MED_ROOT, BIN_MASK_MED_ROOT


PROJ_BASES = './projects/bases'

def prepare_visor_medium_kitchen(pid, suffix='visor_medium', 
                                 is_epic100=False):
    """
    Generate images and masks under 
        `projects/bases/<pid>_visor_medium/images`
        `projects/bases/<pid>_visor_medium/masks`
    """
    src_img_root = IMG_MED_ROOT  # (854 x 480)
    src_mask_root = BIN_MASK_MED_ROOT  # (854 x 480)
    src_list, vf_ids = retrieve_kitchen_frames(
        pid, visor_img_root=src_img_root, is_epic100=is_epic100)
    root = osp.join(PROJ_BASES, f'{pid}_{suffix}')
    images = osp.join(root, 'images')
    masks = osp.join(root, 'masks')
    os.makedirs(images, exist_ok=True)
    os.makedirs(masks, exist_ok=True)

    for src_img, vf_id in zip(src_list, vf_ids):
        name = f'frame_{vf_id}'
        vid, fid = revert_from_VFid(vf_id)
        src_mask = osp.join(src_mask_root, vid, f'{vid}_frame_{fid}.png')
        src_img = osp.abspath(src_img)
        src_mask = osp.abspath(src_mask)
        assert osp.exists(src_img)
        assert osp.exists(src_mask)
        dst_img = osp.join(images, name + '.jpg')
        dst_mask = osp.join(masks, name + '.jpg.png')
        os.link(src_img, dst_img)
        os.link(src_mask, dst_mask)
        # print(src_img, dst_img)
        # print(src_mask, dst_mask)


def get_pid_stat(pid, return_vids=False):
    src_img_root = IMG_MED_ROOT  # (854 x 480)
    src_list, vf_ids = retrieve_kitchen_frames(
        pid, visor_img_root=src_img_root, is_epic100=False)
    if return_vids:
        vids_50 = set([v.split('/')[7] for v in src_list])
    n_50 = len(src_list)
    src_list, vf_ids = retrieve_kitchen_frames(
        pid, visor_img_root=src_img_root, is_epic100=True)
    n_100 = len(src_list)
    if return_vids:
        vids_100 = set([v.split('/')[7] for v in src_list])
        return n_50, n_100, vids_50, vids_100
    return n_50, n_100


if __name__ == '__main__':
    pids = ['P01', 'P02', 'P09', 'P13', 'P25', 'P27', 'P28']
    for pid in pids:
        n50, n100, vids_50, vids_100 = get_pid_stat(pid, return_vids=True)
        print(f"{pid} at EPIC50:\t{n50}, {vids_50}")
        print(f"{pid} at EPIC100:\t{n100}, {vids_100}")

    # pids = ['P02', 'P25', 'P28']
    # for pid in pids:
    #     prepare_visor_medium_kitchen(pid, is_epic100=True)
    # prepare_visor_medium_kitchen('P27', is_epic100=True)