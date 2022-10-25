from pathlib import Path

DATA = Path('./visor_data')
IMG_ROOT = DATA/'sparse_images'
MASK_ROOT = DATA/'sparse_masks'
PROJ_ROOT = Path('./projects')

IMAGEREADER = 'ImageReader'
SIFTEXTRACTION = 'SiftExtraction'
SIFTMATCHING = 'SiftMatching'