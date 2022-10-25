from pathlib import Path

DATA = Path('./visor_data')
IMG_ROOT = DATA/'sparse_images'
MASK_ROOT = DATA/'sparse_masks'
PROJ_ROOT = Path('./projects')

VOCAB_32K = Path('./vocab_bins/vocab_tree_flickr100K_words32K.bin')
VOCAB_256K = Path('./vocab_bins/vocab_tree_flickr100K_words256K.bin')
VOCAB_1M = Path('./vocab_bins/vocab_tree_flickr100K_words1M.bin')

IMAGEREADER = 'ImageReader'
SIFTEXTRACTION = 'SiftExtraction'
SIFTMATCHING = 'SiftMatching'