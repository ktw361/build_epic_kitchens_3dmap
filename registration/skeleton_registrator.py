"""Description
This registrator take two skeletons, the first one as base, and register the second one into the first one.
Each skeleton is a (camera, image, points) triple, the database.db of the first skeleton will be copied.

Resource required:
the first database.db of the first skeleton will be copied (~100 MB)

Implementation:
random num_reg_images will be selected (e.g. arbitrary number 3~100)

Output structure:
<path-to-outdir>/
    database.db
    first_model/  -- check if we really need to copy
    model/
    <kitchen-id>.json

note the database.db and model/ are redundant after <kitchen-id>.json is obtained,
hence they can be deleted to reduce space usage.
Note about EPIC-50 and EPIC-100, so we name kid as P04A (50) or P04B (100)
Each <kid>.json contains:
[
    {
        "model_vid": "P04_01",
        "rot": 3x3 rotation matrix, applying to col-vec
        "transl": 3x1 translation vector,
        "scale": float,
    },
]


Usage:
skeleton_registrator = SkeletonRegistrator(
    first_db_path, first_model_path, first_vid=None,
    out_dir, rgb_frames_root)  # e.g. out_dir=projects/registration/P04A
skeleton_registrator.add_new(
    second_model_path, second_vid, num_reg_images)
skeleton_registrator.add_new(
    third_model_path, third_vid, num_reg_images)
skeleton_registrator.export_transforms()

Script Usage:
- Step1:
    Generate <kitchen-id>.json
    Creating <kitchen-id>/ directory
    Copying to <vid>/{images/, image_list.txt}
    Copying database.db

- Step2:
    Run actual colmap subprocess

- Step3:
    Compute (s, R, t) from registered colmap model
"""

import os
import sys
import os.path as osp
import glob
import numpy as np
import re
import shutil
import subprocess
import argparse
import logging
import tqdm

from registration.functions import (
    extract_common_images, umeyama_ransac, write_registration
)

from colmap_converter.colmap_utils import read_images_binary
from libzhifan import io


class SkeletonRegistrator:

    def __init__(self,
                 out_dir: str,
                 verbose=False):
        self.out_dir = out_dir
        self.db_path = osp.join(self.out_dir, 'database.db')
        self.model_path = osp.join(self.out_dir, 'model')
        assert osp.exists(self.out_dir)

        self.verbose = verbose
        if osp.exists(osp.join(self.out_dir, 'log.txt')):
            os.remove(osp.join(self.out_dir, 'log.txt'))
        if self.verbose:
            self.log_fd = sys.stdout
        else:
            self.log_fd = open(osp.join(out_dir, 'log.txt'), 'w')
        logging.basicConfig(
            format='[%(asctime)s %(levelname)-8s] %(message)s',
            stream=self.log_fd,
            level=logging.DEBUG,
            datefmt='%m/%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def add_new(self, model_path: str, model_vid: str, num_reg_images: int):
        """ This function will trigger colmap to do the follow
        ```
        colmap feature_extractor \
            --database_path $PROJECT_PATH/database.db \
            --image_path $PROJECT_PATH/images \
            --image_list_path /path/to/image-list.txt \
            --ImageReader.existing_camera_id 1

        # Indexing will take a while, depending on the size of existing database
        # e.g. 4k images 4mins
        colmap vocab_tree_matcher \
            --database_path $PROJECT_PATH/database.db \
            --VocabTreeMatching.vocab_tree_path /path/to/vocab-tree.bin \
            --VocabTreeMatching.match_list_path /path/to/image-list.txt

        # use mapper instead of image_registrator + bundle_adjuster for more accurate results
        # The Global Bundle Adjustment takes long (20mins)
        colmap mapper \
            --database_path $PROJECT_PATH/database.db \
            --image_path $PROJECT_PATH/images \
            --input_path /path/to/existing-model \
            --output_path /path/to/model-with-new-images
        ```
        """
        # TODO: feature_extractor, what about camera focal_length?
        new_model_dir = osp.join(self.out_dir, model_vid)  # e.g. P04_01/
        image_list_txt = osp.join(new_model_dir, 'image_list.txt')
        images_dir = osp.join(new_model_dir, 'images')

        # Feature extraction
        commands = [
            'colmap', 'feature_extractor',
            '--database_path', f'{self.db_path}',
            '--image_path', f'{images_dir}',
            '--ImageReader.existing_camera_id', '1'
        ]
        self.logger.info("Running: " + ' '.join(commands))
        proc = subprocess.run(
            commands,
            stdout=self.log_fd,
            stderr=self.log_fd,
            check=True, text=True)
        print(proc)

        # vocab_tree_matcher
        vocab_tree_path = '/home/skynet/Zhifan/build_kitchens_3dmap/vocab_bins/vocab_tree_flickr100K_words256K.bin'
        commands = [
            'colmap', 'vocab_tree_matcher',
            '--database_path', f'{self.db_path}',
            '--VocabTreeMatching.vocab_tree_path', f'{vocab_tree_path}',
            '--VocabTreeMatching.match_list_path', f'{image_list_txt}'
        ]
        self.logger.info("Running: " + ' '.join(commands))
        proc = subprocess.run(
            commands,
            stdout=self.log_fd,
            stderr=self.log_fd,
            check=True, text=True)
        print(proc)

        # mapper (what if we use low-accuracy image_registrator + bundle_adjuster?)
        commands = [
            'colmap', 'mapper',
            '--database_path', f'{self.db_path}',
            '--image_path', f'{images_dir}',
            '--input_path', f'{self.model_path}',
            '--output_path', f'{new_model_dir}'
        ]
        self.logger.info("Running: " + ' '.join(commands))
        proc = subprocess.run(
            commands,
            stdout=self.log_fd,
            stderr=self.log_fd,
            check=True, text=True)
        print(proc)


def generate_infile(first_vid,
                    skeleton_root="/home/skynet/Zhifan/epic_fields_full/skeletons",
                    colmap_out_root="./projects/registration/",
                    settings_save_dir='json_files/registration/input/') -> dict:
    """ Step 1 infile generation
    """
    if len(first_vid) == 6:
        kitchen = first_vid[:3] + 'A'
        pattern = re.compile(first_vid[:4] + '[0-9]{2}_low')
    else:
        kitchen = first_vid[:3] + 'B'
        pattern = re.compile(first_vid[:4] + '[0-9]{3}_low')

    # ktichen=P04A
    out_dir = osp.join(colmap_out_root, kitchen)
    export_path = osp.join(out_dir, f'{kitchen}.json')
    num_reg_images = 100
    ransac = dict(
        max_iterations=100,
        error_thresh=0.5,
        min_inliers=60,
    )
    skeleton_format = osp.join(skeleton_root, f'%s_low')
    model_format = osp.join(skeleton_format, 'sparse/0')
    first_skeleton = skeleton_format % first_vid
    database_path = glob.glob(osp.join(first_skeleton, '*.db'))[0]
    first = dict(
        vid=first_vid,
        model=model_format % first_vid,
        database=database_path)

    skeletons = sorted(
        [osp.join(v, 'sparse/0')
         for v in glob.glob(osp.join(skeleton_root, '*'))
         if pattern.search(v) and first_vid not in v])

    settings = dict(
        out_dir=out_dir,
        num_reg_images=num_reg_images,
        ransac=ransac,
        export_path=export_path,
        first=first,
        skeletons=skeletons)
    io.write_json(
        settings,
        osp.join(settings_save_dir, f'{kitchen}_in.json'), indent=2)
    return settings


def prepare_input(settings):

    def prepare_images(out_dir: str,
                       model_path: str,
                       model_vid: str,
                       num_reg_images: int,
                       rgb_frames_root: str = '/media/skynet/DATA/Datasets/epic-100/rgb'):
        """ This function will prepare the images for registration

        Returns:
            f: image_list.txt
            d: images/
            selected: list of selected images, frame_%010d.jpg
        """
        # Preparation step
        # Note, we need to rename the images! As images in different dir will collide the same name
        # Hence the safest thing is to copy(hard-link) to the designated name (i.e. appending P04_01_ to the prefix)
        new_model_dir = osp.join(out_dir, model_vid)  # e.g. P04_01/
        os.makedirs(new_model_dir, exist_ok=True)
        images = osp.join(model_path, 'images.bin')
        images = read_images_binary(images)
        np.random.seed(0)
        if len(images.keys()) < num_reg_images:
            # We Can't proceed!
            err_msg = f"registered_images {len(images.keys())} < num_reg_images({num_reg_images})"
            return None, None, err_msg
        ids = np.random.choice(list(images.keys()), size=num_reg_images, replace=False)
        selected = [images[i].name for i in ids]
        src_dir = osp.join(rgb_frames_root, model_vid[:3], model_vid)
        d = osp.join(new_model_dir, 'images')
        f = osp.join(new_model_dir, 'image_list.txt')
        os.makedirs(d, exist_ok=True)
        for frame in selected:
            src = osp.join(src_dir, frame)
            dst = f'{model_vid}_{frame}'
            dst = osp.join(d, dst)
            shutil.copy(src, dst)
        with open(f, 'w') as fp:
            fp.writelines('\n'.join([f'{model_vid}_{v}' for v in selected]))
        return f, d, selected

    first_db_path = settings['first']['database']
    first_model_path = settings['first']['model']
    out_dir = settings['out_dir']
    db_path = osp.join(out_dir, 'database.db')
    model_path = osp.join(out_dir, 'model')
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(first_db_path, db_path)
    shutil.copytree(first_model_path, model_path, dirs_exist_ok=True)

    for skeleton_model_path in tqdm.tqdm(settings['skeletons']):
        vid = re.search('P\d{2}_\d{2,3}', skeleton_model_path)[0]
        r, _, err_msg = prepare_images(out_dir, skeleton_model_path, vid,
                           settings['num_reg_images'])
        if r is None:
            print(f"Can't do {vid}, {err_msg}")


def main(args):
    step = args.step

    if step == 1:
        kitchen_id = re.search('P\d{2}[A,B]', args.infile)[0]
        if kitchen_id is None:
            raise ValueError("Can't find kitchen_id from infile")
        if kitchen_id[-1] == 'A':
            first_vid = kitchen_id[:3] + '_01'
        elif kitchen_id[-1] == 'B':
            first_vid = kitchen_id[:3] + '_101'
        settings = generate_infile(first_vid)
        print("Preparing input...")
        prepare_input(settings)

    elif step == 2:
        settings = io.read_json(args.infile)
        out_dir = settings["out_dir"]
        num_reg_images = settings["num_reg_images"]

        skeleton_registrator = SkeletonRegistrator(
            out_dir=out_dir, verbose=args.verbose)

        for skeleton_model_path in settings["skeletons"]:
            vid = re.search('P\d{2}_\d{2,3}', skeleton_model_path)[0]
            print("Adding new model: ", vid)
            if osp.exists(osp.join(out_dir, vid, 'points3D.bin')):
                print("Already registered, skipping")
                continue
            skeleton_registrator.add_new(
                model_path=skeleton_model_path,
                model_vid=vid,
                num_reg_images=num_reg_images)

    elif step == 3:
        settings = io.read_json(args.infile)

        out_dir = settings["out_dir"]
        first_vid = settings["first"]["vid"]
        export_path = settings["export_path"]
        ransac_params  = settings["ransac"]
        max_iterations = ransac_params["max_iterations"]
        error_thresh = ransac_params["error_thresh"]
        min_inliers = ransac_params["min_inliers"]
        model_dof = 3

        write_registration(
            export_path, model_vid=first_vid, s=1.0, R=np.eye(3), t=np.ones(3))

        for skeleton_model_path in settings["skeletons"]:
            vid = re.search('P\d{2}_\d{2,3}', skeleton_model_path)[0]
            _, imgs_dst, imgs_src = extract_common_images(
                out_dir=out_dir, model_path=skeleton_model_path, model_vid=vid)
            if imgs_src.shape[0] < model_dof:
                print(f"Skipping {vid} due to insufficient images {imgs_src.shape[0]}")
                continue
            s, R, t, all_errs = umeyama_ransac(
                imgs_src, imgs_dst,
                k=max_iterations, t=error_thresh, d=min_inliers,
                n=model_dof)
            if s is None:
                print(f"Skipping {vid} due to No Good Models")
                continue
            write_registration(export_path, model_vid=vid, s=s, R=R, t=t)

    else:
        raise ValueError()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str)
    parser.add_argument('--step', type=int, choices=[1, 2, 3])
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
