from typing import List
import os
from pathlib import Path
import shutil
import subprocess

from lib.config_utils import read_ini, write_ini

DATA = Path('./visor_data')
IMG_ROOT = DATA/'sparse_images'
MASK_ROOT = DATA/'sparse_masks'
PROJ_ROOT = Path('./colmap_projects')

_ImageReader = 'ImageReader'
_SiftExtraction = 'SiftExtraction'
_SiftMatching = 'SiftMatching'

class Runner:
    
    def __init__(self, 
                 vid: str, 
                 init=False,
                 init_from='configs/custom.ini',
                 proj_name=None,
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
        self.mask_path = MASK_ROOT/f'{self.vid}'
        self.sparse_dir = proj_dir/'sparse'
        self.dense_dir = proj_dir/'dense'
        self.undistorted_dir = proj_dir/'undistorted'
        self.verbose = verbose

        # configs
        self.camera_model = 'SIMPLE_PINHOLE'

        if init:
            self.setup_project(init_from=init_from)
        self.cfg = read_ini(self.proj_file)

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
        self.cfg[_ImageReader]['camera_model'] = self.camera_model
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
        ]
        commands += self._pack_section_arguments([_ImageReader, _SiftExtraction])
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
        ]
        commands += self._pack_section_arguments([_SiftMatching])
        if self.verbose:
            print(' '.join(commands))
        proc = subprocess.run(
            commands, 
            stdout=None if self.verbose else subprocess.PIPE, 
            check=True, text=True)
        print(proc.stdout)
    
    def mapper_run(self):
        """
        This step is usually slowest.
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
        This step also takes very long.
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

    def auto_reconstruct(self):
        """ 
        The follows the automatic reconstruction command.
        See https://colmap.github.io/cli.html
        """
        raise NotImplementedError
        self.extract_feature()
        self.exhaustive_matcher()
        self.mapper_run()
        self.image_undistorter()
        self.patch_match_stereo()
        self.stereo_fusion()
        self.poinsson_mesher()  # self.delaunay_mesher()


def main():
    # runner = Runner('P01_01', init=True, verbose=True)
    # runner.extract_feature()

    runner = Runner('P01_01', init=False, verbose=True)
    # runner.sequential_matching()
    # runner.mapper_run()
    # runner.image_undistorter()

if __name__ == '__main__':
    main()