
import os
from glob import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import time

from .lib import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
    )
    parser.add_argument(
        "--dir_dst",
        type=str,
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
VID=P01_01; 
DIR_SRC=/work/vadim/datasets/visor/2v6cgv1x04ol22qp9rm9x2j6a7/EPIC-KITCHENS-frames/tar/; 
python -m frame_similarity_filter.filter \
    --src=$DIR_SRC/$VID.tar \
    --dir_dst=filtered/$VID
"""
if __name__ == '__main__':
    args = parse_args()

    # src = f'/work/vadim/datasets/visor/2v6cgv1x04ol22qp9rm9x2j6a7/' + \
    # 'EPIC-KITCHENS-frames/rgb_frames/P28_05'
    # args.src = '/work/vadim/datasets/visor/2v6cgv1x04ol22qp9rm9x2j6a7/' + \
    # 'EPIC-KITCHENS-frames/tar/P28_05.tar'
    # args.dir_dst = 'debug'
    # args.frame_range_max = 1000
    # print('---')
    # print('TODO: QUICK FIX FOR "DARK"/UNINFORMATIVE FRAMES')
    # print('---')
    # args.frame_range_min = 1000

    homographies = make_homography_loader(args)

    graph = calc_graph(homographies, **vars(args))

    fpaths_filtered = graph2fpaths(graph)

    save(fpaths_filtered, args)
