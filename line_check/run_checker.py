import os
import re
import tqdm
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from moviepy import editor

from line_check.checker import LineChecker


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--radius', type=float, default=0.2)
    parser.add_argument('--out-name', type=str, help='example: P01_01-homo')
    parser.add_argument('--fps', type=int, default=15)
    return parser.parse_args()


def main(args):
    radius = args.radius
    out_base = 'outputs/line_check/'
    out_name = args.out_name
    out_dir = os.path.join(out_base, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    anno_points = [
        4.63313, -0.2672, 2.55641,
        -5.22596, 0.352575, 3.04684,
        0.675789, -0.0019428, 2.77022
    ]
    anno_points = np.asarray(anno_points).reshape(-1, 3)

    checker = LineChecker(args.model_dir,
        anno_points=anno_points)
    _ = checker.aggregate(radius=radius, return_dict=True)

    vid = re.search('P\d{2}_\d{2,3}', checker.example_data[0].name)[0]
    fmt = os.path.join(out_dir, '{}')

    for i in tqdm.tqdm(checker.ordered_image_ids):
        name = checker.images[i].name
        img = checker.visualize_compare(i)
        frame = os.path.basename(name)
        Image.fromarray(img).save(fmt.format(frame))

    clip = editor.ImageSequenceClip(sequence=out_dir, fps=args.fps)
    video_file = os.path.join(out_dir, f'{out_name}-fps{args.fps}.mp4')
    clip.write_videofile(video_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)