import json
import os.path as osp
from argparse import ArgumentParser
import numpy as np

from line_check.lite_checker import JsonMultiLineChecker
from libzhifan import io


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile', type=str, help="path to transforms.json")
    parser.add_argument('--line-data', type=str, required=True)
    parser.add_argument('--model_prefix', default='projects/json_models/')
    parser.add_argument('--model_suffix', default='_skeletons.json')
    parser.add_argument(
        '--epic_rgb_root', type=str, default='/media/skynet/DATA/Datasets/epic-100/rgb',
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
    epic_rgb_root = args.epic_rgb_root
    fname_nosuffix = osp.basename(args.infile).split('.')[0]
    out_base = f'outputs/cross_line_check/{fname_nosuffix}'
    model_format = f'{args.model_prefix}%s{args.model_suffix}'  # model_format % vid

    with open(args.infile, 'r') as fp:
        model_infos = json.load(fp)

    # Get line drawn in common reference coordinate
    COMMON = 0
    line = np.asarray(io.read_json(args.line_data)).reshape(-1, 3)
    rot, transl, scale = read_transformation(model_infos[COMMON])
    common_line = line * scale @ rot.T + transl  # In fact identity

    for ind, model_info in enumerate(model_infos):
        vid = model_info['model_vid']
        model_path = model_format % vid
        frames_root = osp.join(epic_rgb_root, vid[:3], vid)
        model = io.read_json(model_path)

        # Take the ref line in ref coordinate, and (inversely) transform it to current model
        rot, transl, scale = read_transformation(model_info)
        common_line_transformed = inverse_transform_line(rot, transl, scale, common_line)
        lines_list = [common_line_transformed]
        line_colors = ['blue']

        checker = JsonMultiLineChecker(
            model['camera'], model['images'],
            anno_points_list=lines_list,
            line_colors=line_colors,
            frames_root=frames_root)
        print(f"Writing model: {vid}")
        checker.write_mp4(radius=None, out_name=vid, fps=args.fps, out_base=out_base)


if __name__ == '__main__':
    args = parse_args()
    main(args)
