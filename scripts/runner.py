from typing import List
import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import os
import os.path as osp
from pathlib import Path
from PIL import Image
import shutil
import subprocess

from lib.config_utils import read_ini, write_ini
from lib.constants import (
    IMAGEREADER, SIFTEXTRACTION,
    SIFTMATCHING, VOCABTREEMATCHING, SEQUENTIALMATCHING,
    MAPPER,
)
from lib.constants import VOCAB_32K, VOCAB_256K, VOCAB_1M


""" Usage:
python scripts/runner.py  \
    image_list_path=sampling/txt/P01/image_list.txt
    mask_mapping_path=sampling/txt/P01/mapping.txt
    matcher=SEQ-32K \
    proj_name=P01-sample-v1
"""


class Runner:

    def __init__(self,
                 cfg: DictConfig,

                 matcher: str,
                 vocab_tree: str,

                 init=False,
                 init_from='configs/custom.ini',
                 verbose=False):
        """
        Args:
            matcher: one of {'vocab', 'seq'}
            vocab_tree: one of {'32K', '256K', '1M'}
        """
        proj_dir = Path(os.getcwd())
        self.proj_dir = proj_dir  # e.g. './colmap_projects/P01_103'

        # Process images path
        self.image_path = to_absolute_path(cfg.images)
        self.image_list_path = self.proj_dir/os.path.basename(cfg.image_list_path)
        self.mask_mapping_path = self.proj_dir/os.path.basename(cfg.mask_mapping_path)
        self.image_list_path = to_absolute_path(self.image_list_path)
        self.mask_mapping_path = to_absolute_path(self.mask_mapping_path)
        shutil.copyfile(to_absolute_path(cfg.image_list_path), self.image_list_path) 
        shutil.copyfile(to_absolute_path(cfg.mask_mapping_path), self.mask_mapping_path) 

        self.proj_file = proj_dir/'project.ini'
        self.database_path = proj_dir/'database.db'
        self.matcher = matcher
        self.hierarchical_mapper = cfg.hierarchical_mapper
        if vocab_tree == '32K':
            self.vocab_tree = VOCAB_32K
        elif vocab_tree == '256K':
            self.vocab_tree = VOCAB_256K
        elif vocab_tree == '1M':
            self.vocab_tree = VOCAB_1M
        else:
            self.vocab_tree = None
        self.sparse_dir = proj_dir/'sparse'
        self.tri_ba_dir = proj_dir/'tri_ba'  # Point Triangulator and Bundle Adjust

        # Logger related
        self.colmap_log_fd = open(proj_dir/'colmap_output.log', 'w')
        logging.basicConfig(filename=proj_dir/'runner.log',
                            filemode='w',
                            level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

        # configs
        self.camera_model = cfg.camera_model

        if init:
            self.setup_project(init_from=to_absolute_path(init_from))
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
        self.logger.info("extract_feature() Started")
        commands = [
            'colmap', 'feature_extractor',
            '--database_path', f'{self.database_path}',
            '--image_path', f'{self.image_path}',
            '--image_list_path', f'{self.image_list_path}',
            '--ImageReader.mask_mapping_path', f'{self.mask_mapping_path}',
        ]

        commands += self._pack_section_arguments([IMAGEREADER, SIFTEXTRACTION])
        print(' '.join(commands), file=self.colmap_log_fd)

        proc = subprocess.run(
            commands,
            stdout=None if self.verbose else self.colmap_log_fd,
            stderr=None if self.verbose else self.colmap_log_fd,
            check=True, text=True)
        if self.verbose:
            print(' '.join(commands))
            print(proc.stdout)
            print(proc.stderr)

    def sequential_matching(self):
        self.logger.info("sequential_matching() Started")
        commands = [
            'colmap', 'sequential_matcher',
            '--database_path', f'{self.database_path}',
        ]
        if self.vocab_tree is not None:
            commands += ['--SequentialMatching.vocab_tree_path',
                         f'{self.vocab_tree}']
        commands += self._pack_section_arguments([SIFTMATCHING])
        commands += self._pack_section_arguments([SEQUENTIALMATCHING])
        print(' '.join(commands), file=self.colmap_log_fd)
        proc = subprocess.run(
            commands,
            stdout=None if self.verbose else self.colmap_log_fd,
            stderr=None if self.verbose else self.colmap_log_fd,
            check=True, text=True)
        if self.verbose:
            print(' '.join(commands))
            print(proc.stdout)

    def vocab_matching(self):
        self.logger.info("vocab_matching() Started")
        commands = [
            'colmap', 'vocab_tree_matcher',
            '--database_path', f'{self.database_path}',
            '--VocabTreeMatching.vocab_tree_path', f'{self.vocab_tree}',
        ]
        commands += self._pack_section_arguments([SIFTMATCHING])
        commands += self._pack_section_arguments([VOCABTREEMATCHING])
        print(' '.join(commands), file=self.colmap_log_fd)
        proc = subprocess.run(
            commands,
            stdout=None if self.verbose else self.colmap_log_fd,
            stderr=None if self.verbose else self.colmap_log_fd,
            check=True, text=True)
        if self.verbose:
            print(' '.join(commands))
            print(proc.stdout)
            print(proc.stderr)

    def mapper_run(self, hierarchical=False):
        """
        This step is usually slow.

        Outputs in `proj_dir/sparse/0`
        """
        self.logger.info("mapper_run() Started")
        os.makedirs(self.sparse_dir, exist_ok=True)
        mapper = 'mapper' if not hierarchical else 'hierarchical_mapper'
        commands = [
            'colmap', mapper,
            '--database_path', f'{self.database_path}',
            '--image_path', f'{self.image_path}',
            '--output_path', f'{self.sparse_dir}',
        ]
        commands += self._pack_section_arguments([MAPPER])
        print(' '.join(commands), file=self.colmap_log_fd)
        proc = subprocess.run(
            commands,
            stdout=None if self.verbose else self.colmap_log_fd,
            stderr=None if self.verbose else self.colmap_log_fd,
            check=True, text=True)
        if self.verbose:
            print(' '.join(commands))
            print(proc.stdout)
            print(proc.stderr)

    def print_summary(self, model_id=0):
        commands = [
            'colmap', 'model_analyzer',
            '--path', f'{self.sparse_dir}/{model_id}',
        ]
        proc = subprocess.run(
            commands,
            stdout=None if self.verbose else self.colmap_log_fd,
            stderr=None if self.verbose else self.colmap_log_fd,
            check=True, text=True)
        if self.verbose:
            print(proc.stdout)
            print(proc.stderr)

    def compute_sparse(self):
        self.logger.info("compute_sparse() Started")
        self.extract_feature()
        if self.matcher == 'vocab':
            self.vocab_matching()
        elif self.matcher == 'seq':
            self.sequential_matching()
        self.mapper_run(hierarchical=self.hierarchical_mapper)
        self.print_summary()
        self.logger.info("compute_sparse() Finished")


@hydra.main(config_path='../configs', config_name='runner')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    matcher_raw = cfg.matcher
    if '32K' in matcher_raw:
        vocab_tree = '32K'
    elif '256K' in matcher_raw:
        vocab_tree = '256K'
    elif '1M' in matcher_raw:
        vocab_tree = '1M'
    if 'SEQ' in matcher_raw:
        matcher = 'seq'
    else:
        matcher = 'vocab'

    runner = Runner(
        cfg=cfg,
        matcher=matcher,
        vocab_tree=vocab_tree,
        init=True,
        verbose=cfg.verbose)
    runner.compute_sparse()


if __name__ == '__main__':
    main()
