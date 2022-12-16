
import os
from glob import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import time

from homography_filter.lib import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
    )
    parser.add_argument(
        "--dst_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--overlap",
        # default=0.85,
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--frame_range_min",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--frame_range_max",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--filtering_scale",
        default=1,
        type=int,
    )
    parser.add_argument(
        '-f',
        type=str,
        default=None
    )
    args = parser.parse_args()
    return args


def make_homography_loader(args):

    images = Images(args.src, scale=args.filtering_scale)
    print(f'Found {len(images.imreader.fpaths)} images.')
    features = Features(images)
    matches = Matches(features)
    homographies = Homographies(images, features, matches)

    return homographies


def save(fpaths_filtered, args):
    imreader = ImageReader(src=args.src)
    dir_dst = args.dir_dst
    dir_images = os.path.join(dir_dst, 'images')
    # shutil.rmtree(dir_images)
    os.makedirs(dir_images)
    extract_frames(dir_images, fpaths_filtered, imreader)
    save_as_video(os.path.join(dir_dst, 'video'), fpaths_filtered, imreader)


""" Possible usage:
python homography_filter/filter.py \
    --src ~/data/epic_rgb_frames/P01/P01_09 \
    --dst_file /home/skynet/Zhifan/build_kitchens_3dmap/sampling/txt/P01_09/image_list_homo.txt 
"""
if __name__ == '__main__':
    args = parse_args()

    homographies = make_homography_loader(args)

    graph = calc_graph(homographies, **vars(args))

    fpaths_filtered = graph2fpaths(graph)
    lines = [v+'\n' for v in fpaths_filtered]
    dir_name = os.path.dirname(args.dst_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(args.dst_file, 'w') as fp:
        fp.writelines(lines)

    # io.write_txt(fpaths_filtered, args.dst_file)
    # save(fpaths_filtered, args)
