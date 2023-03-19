
# inrease if line is too small on image, e.g. to 10e6 or so
SZ_LINE = 10e5

from argparse import ArgumentParser
# from tqdm.notebook import tqdm
from tqdm import tqdm
import pycolmap
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from line_check.impl_vadim.vadim import *
import numpy as np
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--anno-path', type=str)
    parser.add_argument('--out-name', type=str, help='example: P01_01-homo')
    parser.add_argument('--frames-root', type=str)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--each_nth', type=int, default=1, help='for debugging, e.g. select only every 50th frame for visualisation.')
    parser.add_argument('-f', type=str)
    return parser.parse_args()


def load_line(json_path):
    line = np.asarray(pd.read_json(json_path)).reshape(-1, 3)
    if len(line) > 2:
        # if we have more 3 or more points annotated, fit line
        line_mean, line_dir = fit_line(line)
    else:
        # otherwise calc direction from 2 points
        line_mean = line.mean(axis=0)
        line_size = np.linalg.norm(line[1] - line[0])
        line_dir = line[1] - line[0]
    return line_mean, line_dir


def main(args):
    ims = []

    colmap_model = pycolmap.Reconstruction(args.model_dir)
    line_mean, line_dir = load_line(args.anno_path)

    for k in tqdm(sorted(list(colmap_model.images))[::args.each_nth]):
    # for k in tqdm(list(range(941, 1156))[:]):

        image = colmap_model.images[k]
        t = image.tvec
        R = image.rotation_matrix()

        im = np.array(Image.open(os.path.join(args.frames_root, image.name)))

        imhw = im.shape[:2]

        # get line center and direction on screen
        line_mean_s = (world2screen(line_mean, R, t, colmap_model, with_radial_dist=False))
        line_dir_s = (world2screen(line_mean + line_dir, R, t, colmap_model, with_radial_dist=False)) - line_mean_s
        line_dir_s_normed = line_dir_s / np.linalg.norm(line_dir_s)

        # calculate line on screen
        line_on_screen = np.array([line_mean_s - line_dir_s_normed * SZ_LINE, line_mean_s + line_dir_s_normed * SZ_LINE])

        # clip line to image height and width
        line_on_screen_clipped = clip_line(imhw, line_on_screen)
        line_on_screen_clipped_x, line_on_screen_clipped_y = line_on_screen_clipped[[1, 3]], line_on_screen_clipped[[0, 2]]

        f = plt.figure(figsize=figsize(im))
        plt.title(''.join(list(filter(lambda x: x.isdigit(), image.name))))
        plt.imshow(im)
        plt.tight_layout()
        plt.plot(line_on_screen_clipped_x, line_on_screen_clipped_y, color='red', linewidth=4)
        plt.axis('off')
        im_ = fig2im(f)
        ims.append(im_)

        img_name = image.name
        save_name = os.path.join('outputs', f'{args.out_name}/{img_name}')
        Image.fromarray(im_).save(save_name)
        # plt.imshow(im_)
        # plt.show()

    return ims


if __name__ == '__main__':

    args = parse_args()
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(f'outputs/{args.out_name}', exist_ok=True)
    ims = main(args)
    write_mp4(f'outputs/{args.out_name}', ims)