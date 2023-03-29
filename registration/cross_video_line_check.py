import os
import re, json
import os.path as osp
from argparse import ArgumentParser
import numpy as np

from lib.base_type import ColmapModel
from line_check.checker import LineCheckerFromModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--radius', type=float, default=0.2)
    parser.add_argument(
        '--epic-rgb-root', type=str, required=True,
        help="directory structured as e.g. <root>/P01/P01_101/frame_%010d.jpg")
    parser.add_argument('--fps', type=int, default=10)
    return parser.parse_args()


def read_transformation(d: dict):
    rot = np.asarray(d['rot']).reshape(3, 3)
    transl = np.asarray(d['transl'])
    scale = d['scale']
    return rot, transl, scale


def inverse_transform_line(rot, transl, scale, line) -> np.ndarray:
    """
    Args:
        rot: (3, 3) applying to col-vec
        transl: (3,)
        scale: float
        line: (2, 3)
    """
    inv_scale = 1 / scale
    inv_rot = rot.T
    inv_transl = - inv_scale * inv_rot @ transl
    return line * inv_scale @ inv_rot.T + inv_transl


def main(args):
    radius = args.radius
    epic_rgb_root = args.epic_rgb_root
    fname_nosuffix = osp.basename(args.infile).split('.')[0]
    out_base = f'outputs/cross_line_check/{fname_nosuffix}'

    with open(args.infile, 'r') as fp:
        model_infos = json.load(fp)

    base_line = None  # base line, represented as points, in base coordinate. (2, 3)
    for ind, model_info in enumerate(model_infos):
        model_path = model_info['model_path']

        vid = re.search('P\d{2}_\d{2,3}', model_path)[0]
        pid = vid.split('_')[0]
        frames_root = osp.join(epic_rgb_root, pid, vid)

        model = ColmapModel(model_path)
        rot, transl, scale = read_transformation(model_info)

        if ind == 0:
            line = np.asarray(model_info['line']).reshape(-1, 3)
            base_line = line * scale @ rot.T + transl

        # Take the base_line in base coordinate, and transform it to single model
        line = inverse_transform_line(rot, transl, scale, base_line)

        checker = LineCheckerFromModel(
            model, anno_points=line, frames_root=frames_root)
        print(f"Writing model: {vid}")
        checker.write_mp4(
            radius=radius, out_name=vid, fps=args.fps, out_base=out_base)


if __name__ == '__main__':
    args = parse_args()
    main(args)
