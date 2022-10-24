import numpy as np
from PIL import Image


def visor_to_colmap_mask(in_path: str, out_path: str) -> np.ndarray:
    mask = np.asarray(Image.open(in_path))
    h, w = mask.shape
    out = np.ones((h, w, 4), dtype=np.uint8) * 255
    out[mask!=0, :] = [0, 0, 0, 255]
    if out_path is not None:
        Image.fromarray(out).save(out_path)