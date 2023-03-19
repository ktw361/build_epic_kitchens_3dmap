import os
import re, json
import tqdm
from argparse import ArgumentParser
import numpy as np
import cv2
from PIL import Image
from moviepy import editor

from line_check.checker import LineChecker


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--radius', type=float, default=0.2)
    parser.add_argument('--anno-path', type=str)
    parser.add_argument('--out-name', type=str, help='example: P01_01-homo')
    parser.add_argument('--frames-root', type=str)
    parser.add_argument('--fps', type=int, default=10)
    return parser.parse_args()


def main(args):
    radius = args.radius
    frames_root = args.frames_root
    out_base = 'outputs/line_check/'
    out_name = args.out_name
    out_dir = os.path.join(out_base, args.out_name)
    os.makedirs(out_dir, exist_ok=True)

    checker = LineChecker(args.model_dir,
        anno_path=args.anno_path, frames_root=frames_root)
    _ = checker.aggregate(radius=radius, return_dict=True)

    # vid = re.search('P\d{2}_\d{2,3}', checker.example_data[0].name)[0]
    fmt = os.path.join(out_dir, '{}')

    for img_id in tqdm.tqdm(checker.ordered_image_ids):
        name = checker.images[img_id].name
        img = checker.visualize_compare(img_id) # , display=display)
        r = checker.report_single(img_id)
        if r[0] != 'COMPUTE':
            text = r[0]
        else:
            text = f'err: {r[1]:.3f}'
        cv2.putText(img, text, (checker.camera.width//3, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        frame_number = re.search('\d{10,}', 
                                 checker.get_image_by_id(img_id).name)[0]
        cv2.putText(img, frame_number, 
                    (checker.camera.width//4, checker.camera.height * 31 // 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        frame = os.path.basename(name)
        Image.fromarray(img).save(fmt.format(frame))

    clip = editor.ImageSequenceClip(sequence=out_dir, fps=args.fps)
    video_file = os.path.join(out_base, f'{out_name}-fps{args.fps}.mp4')
    clip.write_videofile(video_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
