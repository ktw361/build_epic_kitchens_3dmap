"""
Miscellaneous image/mask preprocessing.
"""
import os
from PIL import Image
import tqdm
from multiprocessing import Pool, cpu_count
from lib.constants import (
    IMG_ROOT, IMG_MED_ROOT, MASK_ROOT, BIN_MASK_ROOT, BIN_MASK_MED_ROOT,
)
from lib.utils import visor_to_colmap_mask


"""
tl;dr Resize image from 1920x1080 to 854x480
"""

def generate_image_medium_worker(vid):
    resize = (854, 480)
    src_root = os.path.join(IMG_ROOT, vid)
    dst_root = os.path.join(IMG_MED_ROOT, vid)
    os.makedirs(dst_root, exist_ok=False)
    for name in (os.listdir(src_root)):
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_root, name)
        img = Image.open(src)
        img.resize(resize).save(dst)

def generate_images_medium(mp=False):
    if mp is False:
        resize = (854, 480)
        vids = os.listdir(IMG_ROOT)
        for vid in tqdm.tqdm(vids):
            src_root = os.path.join(IMG_ROOT, vid)
            dst_root = os.path.join(IMG_MED_ROOT, vid)
            os.makedirs(dst_root, exist_ok=False)
            
            for name in tqdm.tqdm(os.listdir(src_root)):
                src = os.path.join(src_root, name)
                dst = os.path.join(dst_root, name)
                img = Image.open(src)
                img.resize(resize).save(dst)
    else:
        with Pool(cpu_count() // 2) as pool:
            vids = os.listdir(IMG_ROOT)
            pool.map(generate_image_medium_worker, vids)


"""
tl;dr Convert colorful mask from 854x480 to binary one, with diff size.
"""

def binarize_worker(vid):
    src_root = os.path.join(MASK_ROOT, vid)
    dst_root = os.path.join(BIN_MASK_MED_ROOT, vid)
    os.makedirs(dst_root, exist_ok=False)
    
    for name in (os.listdir(src_root)):
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_root, name)
        visor_to_colmap_mask(src, dst, resize=None)

def binarize_visor_masks(mp=False):
    """
    Preprocess VISOR colored mask into colmap binary masks,
    then save them on the disk, 
    so that later they can be hard/soft linked into project directory.
    """
    # resize = (1920, 1080)
    vids = os.listdir(MASK_ROOT)
    if mp is True:
        pool = Pool(cpu_count())
        pool.map(binarize_worker, vids)
    else:
        for vid in tqdm.tqdm(vids):
            binarize_worker(vid)


if __name__ == '__main__':
    # generate_images_medium(mp=True)
    binarize_visor_masks(mp=True)