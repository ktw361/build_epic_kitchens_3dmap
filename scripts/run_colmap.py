import tqdm
from typing import List
import os
from pathlib import Path
import shutil
import subprocess

from lib.config_utils import read_ini, write_ini
from lib.utils import visor_to_colmap_mask
from lib.constants import (
    IMG_ROOT, MASK_ROOT, PROJ_ROOT,
    IMAGEREADER, SIFTEXTRACTION, SIFTMATCHING)


class Runner:
    
    def __init__(self, 
                 vid: str, 
                 init=False,
                 init_from='configs/custom.ini',
                 proj_name=None,
                 create_masks=False,
                 verbose=False):
        if proj_name is None:
            proj_dir = PROJ_ROOT/f'{vid}'
        else:
            proj_dir = PROJ_ROOT/f'{proj_name}'
        os.makedirs(proj_dir, exist_ok=True)

        self.vid = vid
        self.proj_dir = proj_dir  # e.g. './colmap_projects/P01_103'
        self.proj_file = proj_dir/'project.ini'
        self.database_path = proj_dir/'database.db'
        self.image_path = IMG_ROOT/f'{self.vid}'
        self.mask_path = proj_dir/'masks'
        self.vocab_tree_path = proj_dir/'vocab_tree_path/vocab_tree_flickr100K_words256K.bin'
        self.sparse_dir = proj_dir/'sparse'
        self.dense_dir = proj_dir/'dense'
        self.undistorted_dir = proj_dir/'undistorted'
        self.stereo_fused = proj_dir/'stereo_fused.ply'
        self.verbose = verbose

        if create_masks:
            self.create_masks(
                mask_src_dir=MASK_ROOT/f'{self.vid}',
                mask_dst_dir=self.mask_path)

        # configs
        self.camera_model = 'SIMPLE_PINHOLE'

        if init:
            self.setup_project(init_from=init_from)
        self.cfg = read_ini(self.proj_file)
    
    def create_masks(self, mask_src_dir, mask_dst_dir):
        """
        Copy & convert masks in visor dir to local {mask_path}
        """
        if not os.path.exists(mask_dst_dir):
            os.makedirs(mask_dst_dir, exist_ok=True)
        print('Copying masks.')
        for mask_path in tqdm.tqdm(os.listdir(mask_src_dir)):
            src = mask_src_dir/mask_path
            dst = mask_dst_dir/(mask_path.replace('.png', '.jpg.png'))
            visor_to_colmap_mask(src, dst)

    def setup_project(self, init_from: str, use_colmap=False):
        if use_colmap:
            commands = [
                'colmap', 'project_generator',
                '--output_path', f'{self.proj_dir}',
            ]
            proc = subprocess.run(
                commands, stdout=subprocess.PIPE, check=True, text=True
            )
            print(proc.stdout)
        else:
            shutil.copyfile(init_from, f'{self.proj_file}')
        
        # setup config and write back
        self.cfg = read_ini(self.proj_file)
        self.cfg['root']['image_path'] = str(self.image_path)
        self.cfg['root']['database_path'] = str(self.database_path)
        self.cfg[IMAGEREADER]['camera_model'] = self.camera_model
        write_ini(self.cfg, self.proj_file)
    
    def _pack_section_arguments(self, sections: List[str]):
        ret_args = []
        for sec in sections:
            for k, v in self.cfg[sec].items():
                argk = f'--{sec}.{k}'
                argv = f'{v}'
                ret_args += [argk, argv]
        return ret_args

    def extract_feature(self):
        commands = [
            'colmap', 'feature_extractor', 
            '--database_path', f'{self.database_path}',
            '--image_path', f'{self.image_path}',
            '--ImageReader.mask_path', f'{self.mask_path}',
        ]
        commands += self._pack_section_arguments([IMAGEREADER, SIFTEXTRACTION])
        if self.verbose:
            print(' '.join(commands))

        proc = subprocess.run(
            commands, 
            stdout=None if self.verbose else subprocess.PIPE, 
            stderr=None if self.verbose else subprocess.PIPE, 
            check=True, text=True)
        if self.verbose:
            print(proc.stdout)
            print(proc.stderr)
    
    def sequential_matching(self):
        commands = [
            'colmap', 'sequential_matcher', 
            '--database_path', f'{self.database_path}',
            '--SequentialMatching.vocab_tree_path', f'{self.vocab_tree_path}',
        ]
        commands += self._pack_section_arguments([SIFTMATCHING])
        if self.verbose:
            print(' '.join(commands))
        proc = subprocess.run(
            commands, 
            stdout=None if self.verbose else subprocess.PIPE, 
            check=True, text=True)
        print(proc.stdout)

    # def vocab_matching(self):
    #     commands = [
    #         'colmap', 'vocab_tree_builder', 
    #         '--database_path', f'{self.database_path}',
    #     ]
    #     # commands += self._pack_section_arguments([_SiftMatching])
    #     if self.verbose:
    #         print(' '.join(commands))
    #     proc = subprocess.run(
    #         commands, 
    #         stdout=None if self.verbose else subprocess.PIPE, 
    #         check=True, text=True)
    #     print(proc.stdout)
    
    def mapper_run(self):
        """
        This step is usually slow.

        Outputs in `proj_dir/sparse/0`
        """
        os.makedirs(self.sparse_dir, exist_ok=True)
        commands = [
            'colmap', 'mapper', 
            '--database_path', f'{self.database_path}',
            '--image_path', f'{self.image_path}',
            '--output_path', f'{self.sparse_dir}',
        ]
        if self.verbose:
            print(' '.join(commands))
        proc = subprocess.run(
            commands, 
            stdout=None if self.verbose else subprocess.PIPE, 
            check=True, text=True)
        print(proc.stdout)
    
    def image_undistorter(self):
        """
        This step will copy lots of images
        """
        input_path = self.sparse_dir/'0'
        commands = [
            'colmap', 'image_undistorter', 
            '--image_path', f'{self.image_path}',
            '--input_path', f'{input_path}',
            '--output_path', f'{self.undistorted_dir}',
            '--copy_policy', 'copy',
        ]
        if self.verbose:
            print(' '.join(commands))
        proc = subprocess.run(
            commands, 
            stdout=None if self.verbose else subprocess.PIPE, 
            stderr=None if self.verbose else subprocess.PIPE, 
            check=True, text=True)
        if self.verbose:
            print(proc.stdout)
            print(proc.stderr)
        
    def patch_match_stereo(self):
        """
        This step also takes very long. (e.g. 1.5h)

        TODO: 
            --geom_consistency true

        Output two dirs:
            `self.undistorted_dir/stereo/{depth_maps,normal_maps}/`
        """
        commands = [
            'colmap', 'patch_match_stereo', 
            '--workspace_path', f'{self.undistorted_dir}',
        ]
        if self.verbose:
            print(' '.join(commands))

        proc = subprocess.run(
            commands, 
            stdout=None if self.verbose else subprocess.PIPE, 
            check=True, text=True)
        print(proc.stdout)

    def stereo_fusion(self):
        """
        Outputs:
            `proj_dir/stereo_fused.ply`
        """
        commands = [
            'colmap', 'stereo_fusion', 
            '--workspace_path', f'{self.undistorted_dir}',
            '--output_path', self.stereo_fused,
        ]
        if self.verbose:
            print(' '.join(commands))

        proc = subprocess.run(
            commands, 
            stdout=None if self.verbose else subprocess.PIPE, 
            check=True, text=True)
        print(proc.stdout)

    def delaunay_mesher(self):
        """
        colmap delaunay_mesher \
            --input_path colmap/dense \
            --output_path delaunay-output.ply

        colmap poisson_mesher \
            --input_path colmap/dense/fused.ply \
            --output_path poisson-output.ply \
        """
        pass

    def compute_sparse(self):
        self.extract_feature()
        self.sequential_matching()
        self.mapper_run()

    def auto_reconstruct(self):
        """ 
        The follows the automatic reconstruction command.
        See https://colmap.github.io/cli.html
        """
        raise NotImplementedError
        self.extract_feature()
        self.sequential_matching()
        self.mapper_run()
        self.image_undistorter()
        self.patch_match_stereo()
        self.stereo_fusion()
        self.delaunay_mesher()  
        self.poinsson_mesher()


def main():
    # runner = Runner('P01_01', init=True, verbose=True)
    runner = Runner('P01_01', 
                    init=True, 
                    verbose=True,
                    # create_masks=True,
                    proj_name='P01_01_v0.2')

    runner.extract_feature()
    runner.sequential_matching()
    # runner.vocab_matching()
    runner.mapper_run()
    # runner.image_undistorter()
    # self.patch_match_stereo()
    # runner.stereo_fusion()

if __name__ == '__main__':
    main()