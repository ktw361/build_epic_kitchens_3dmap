import os
import os.path as osp
from lib.utils import retrieve_kitchen_frames, revert_from_VFid
from lib.constants import IMG_MED_ROOT, BIN_MASK_MED_ROOT, PROJ_BASES


def prepare_visor_medium_kitchen(pid, suffix='visor_medium', is_epic100=False):
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
        dst_mask = osp.join(masks, name + '.png')
        os.symlink(src_img, dst_img)
        os.symlink(src_mask, dst_mask)
        # print(src_img, dst_img)
        # print(src_mask, dst_mask)


def prepare_visor_medium_kitchen_simple_mask(pid, 
                                             suffix='visor_medium_simple_mask',
                                             is_epic100=False):
    """
    Generate images and masks under 
        `projects/bases/<pid>_visor_medium_simple_mask/images`
        `projects/bases/<pid>_visor_medium_simple_mask/masks`
    """
    src_img_root = IMG_MED_ROOT  # (854 x 480)
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
        src_mask = './extra/simple_mask_854x480.png'
        src_img = osp.abspath(src_img)
        src_mask = osp.abspath(src_mask)
        assert osp.exists(src_img)
        assert osp.exists(src_mask)
        dst_img = osp.join(images, name + '.jpg')
        dst_mask = osp.join(masks, name + '.png')
        os.symlink(src_img, dst_img)
        os.symlink(src_mask, dst_mask)
        # print(src_img, dst_img)
        # print(src_mask, dst_mask)


if __name__ == '__main__':
    prepare_visor_medium_kitchen('P01')
    # prepare_visor_medium_kitchen_simple_mask('P01')
    prepare_visor_medium_kitchen('P27', is_epic100=True)
    # prepare_visor_medium_kitchen_simple_mask('P27', is_epic100=True)