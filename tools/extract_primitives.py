from argparse import ArgumentParser
import numpy as np
import tqdm
from colmap_converter.colmap_utils import read_model
from libzhifan import io

""" Extract the primitive information of the colmap output
I.e. extract the
- camera params
- Image w2c qvec + tvec, name
- Points 3D. xyz + rgb
    -rgb in [0, 255]

The image-keypoint correspondence are thrown away
"""

def pythonify(obj):
    if isinstance(obj, list):
        return list(map(pythonify, obj))
    if isinstance(obj, dict):
        return {k: pythonify(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return pythonify(obj.tolist())
    return obj


def extract_primitives(model_dir: str) -> dict:
    """
    Returns:
        {
            'cameras': [xxx],
            'images': [
                [qw, qx, qy, qz, tx, ty, tz, name]  # w2c
            ],
            'points': [
                [x, y, z, r, g, b]  # rgb in 0-255
            ]
        }

        All the array will be converted to naive python list
    """
    cameras, images, points = read_model(model_dir, ext='.bin')
    cameras = pythonify([v._asdict() for v in cameras.values()])
    assert len(cameras) == 1

    print("Parsing images...")
    image_list = []
    for img in tqdm.tqdm(images.values()):
        image_list.append(img.qvec.tolist() + img.tvec.tolist() + [img.name])
    print("Parsing points...")
    point_list = []
    for pt in tqdm.tqdm(points.values()):
        point_list.append(pt.xyz.tolist() + pt.rgb.tolist())

    ret = dict(
        cameras=cameras,
        images=image_list,
        points=point_list)
    return ret


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str,
        help="The directory that contains (cameras.bin, images.bin, points.bin)")
    parser.add_argument('--out_file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_dir = args.model_dir
    out_file = args.out_file
    assert out_file.endswith('.json')
    model = extract_primitives(model_dir)
    io.write_json(model, out_file)
