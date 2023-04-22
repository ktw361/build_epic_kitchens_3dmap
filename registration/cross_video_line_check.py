import json
import os.path as osp
from argparse import ArgumentParser
import numpy as np

from lib.base_type import ColmapModel
from line_check.checker import MultiLineChecker


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--radius', type=float, default=0.2)
    parser.add_argument('--model_prefix', default='projects/')
    parser.add_argument('--model_suffix', default='')
    parser.add_argument(
        '--epic_rgb_root', type=str, required=True,
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
    model_format = f'{args.model_prefix}%s{args.model_suffix}'  # model_format % vid

    with open(args.infile, 'r') as fp:
        model_infos = json.load(fp)

    # Get line drawn in common reference coordinate
    COMMOND_INDEX = 0
    line = np.asarray(model_infos[COMMOND_INDEX]['line']).reshape(-1, 3)
    rot, transl, scale = read_transformation(model_infos[COMMOND_INDEX])
    common_line = line * scale @ rot.T + transl

    for ind, model_info in enumerate(model_infos):
        vid = model_info['model_vid']
        model_path = model_format % vid

        frames_root = osp.join(epic_rgb_root, vid[:3], vid)

        model = ColmapModel(model_path)
        rot, transl, scale = read_transformation(model_info)

        # Take the common_line in common coordinate, and transform it to single model
        common_line_transformed = inverse_transform_line(rot, transl, scale, common_line)
        lines_list = [common_line_transformed]
        line_colors = ['blue']

        if 'line' in model_info:
            line = np.asarray(model_info['line']).reshape(-1, 3)
            lines_list.insert(0, line)
            line_colors.insert(0, 'yellow')

        checker = MultiLineChecker(
            model,
            anno_points_list=lines_list,
            line_colors=line_colors,
            frames_root=frames_root)
        print(f"Writing model: {vid}")
        checker.write_mp4(
            radius=radius, out_name=vid, fps=args.fps, out_base=out_base)


if __name__ == '__main__':
    args = parse_args()
    main(args)
