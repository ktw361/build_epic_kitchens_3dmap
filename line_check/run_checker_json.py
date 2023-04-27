import os
import re
from argparse import ArgumentParser
import numpy as np

from line_check.lite_checker import JsonMultiLineChecker
from libzhifan import io


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--out-name', type=str, help='example: P01_01-homo', required=True)
    parser.add_argument('--epic_rgb_root', type=str, 
                        default='/media/skynet/DATA/Datasets/epic-100/rgb')
    parser.add_argument('--fps', type=int, default=10)
    return parser.parse_args()


def main(args):
    epic_rgb_root = args.epic_rgb_root
    vid = re.search('P\d{2}_\d{2,3}', args.infile)[0]
    frames_root = os.path.join(epic_rgb_root, vid[:3], vid)
    out_base = 'outputs/line_check/'
    out_name = args.out_name
    out_dir = os.path.join(out_base, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    model = io.read_json(args.infile)
    line = np.asarray(model['line']).reshape(-1, 3)
    checker = JsonMultiLineChecker(
        model['cameras'][0], model['images'],
        anno_points_list=[line],
        line_colors=['yellow'],
        frames_root=frames_root)
    checker.write_mp4(
        radius=None, out_name=out_name, fps=args.fps, out_base=out_base)


if __name__ == '__main__':
    args = parse_args()
    main(args)
