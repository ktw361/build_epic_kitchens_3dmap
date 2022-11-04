from pathlib import Path
from hydra.utils import to_absolute_path

DATA = Path(to_absolute_path('./visor_data'))

IMG_ROOT = DATA/'sparse_images'  # (1920 x 1080)
IMG_MED_ROOT = DATA/'sparse_images_medium'  # (854 x 480)

MASK_ROOT = DATA/'sparse_masks'  # Note: colored
BIN_MASK_ROOT = DATA/'sparse_binary_masks'
BIN_MASK_MED_ROOT = DATA/'sparse_binary_masks_medium'

VOCAB_32K   = Path(to_absolute_path('./vocab_bins/vocab_tree_flickr100K_words32K.bin'))
VOCAB_256K  = Path(to_absolute_path('./vocab_bins/vocab_tree_flickr100K_words256K.bin'))
VOCAB_1M    = Path(to_absolute_path('./vocab_bins/vocab_tree_flickr100K_words1M.bin'))

IMAGEREADER = 'ImageReader'
SIFTEXTRACTION = 'SiftExtraction'
SIFTMATCHING = 'SiftMatching'
SEQUENTIALMATCHING = 'SequentialMatching'
VOCABTREEMATCHING = 'VocabTreeMatching'
MAPPER = 'Mapper'