import argparse
import os.path as osp
import re
import numpy as np
from libzhifan import io

from lib.base_type import ColmapModel
from registration.functions import (
    get_common_frames, umeyama_ransac, write_registration
)


def extract_common_images(out_dir, model_path: str, model_vid: str):
    register_result = ColmapModel(osp.join(out_dir, model_vid))
    origin = ColmapModel(model_path)
    selected = io.read_txt(osp.join(out_dir, model_vid, 'image_list.txt'))
    selected = [v[0] for v in selected]
    frames_reg = selected
    frames_ori = [re.search('frame_\d{10}.jpg', v)[0] for v in frames_reg]
    common, imgs_dst, imgs_src = get_common_frames(
        register_result, origin, frames_reg, frames_ori, return_pos=True)
    return common, imgs_dst, imgs_src


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str)
    return parser.parse_args()


def main(args):
    settings = io.read_json(args.infile)
    
    out_dir = settings["out_dir"]
    first_vid = settings["first"]["vid"]
    export_path = settings["export_path"]
    ransac_params  = settings["ransac"]
    max_iterations = ransac_params["max_iterations"]
    error_threshold = ransac_params["error_threshold"]
    min_inliers = ransac_params["min_inliers"]

    write_registration(
        export_path, model_vid=first_vid, s=1.0, R=np.eye(3), t=np.ones(3))

    for skeleton_model_path in settings["skeletons"]:
        vid = re.search('P\d{2}_\d{2,3}', skeleton_model_path)
        common, imgs_dst, imgs_src = extract_common_images(
            out_dir=out_dir, model_path=skeleton_model_path, model_vid=vid)
        s, R, t, all_errs = umeyama_ransac(
            imgs_src, imgs_dst, imgs_src, 
            k=max_iterations, t=error_threshold, n=min_inliers)
        write_registration(export_path, model_vid=vid, s=s, R=R, t=t)