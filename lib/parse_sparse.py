from typing import NamedTuple
from functools import cached_property
from argparse import ArgumentParser
import os
import numpy as np
import sqlite3
import pandas as pd
from colmap_converter import colmap_utils
from libzhifan.geometry import create_pcd_scene


class ColmapDB(NamedTuple):
    cameras: pd.DataFrame
    descriptors: pd.DataFrame
    images: pd.DataFrame
    keypoints: pd.DataFrame
    matches: pd.DataFrame
    two_view_geometries: pd.DataFrame

class SparseProj:
    
    def __init__(self, proj_root, model_id=0):
        prefix = f'{proj_root}/sparse/{model_id}'
        self.database_path = f'{proj_root}/database.db'
        self.prefix = prefix

        def _safe_read(path, func):
            if not os.path.exists(path):
                return None
            as_list = lambda x: list(dict(x).values())
            return as_list(func(path))
        self.cameras_raw = _safe_read(f'{prefix}/cameras.bin', colmap_utils.read_cameras_binary)
        self.points_raw  = _safe_read(f'{prefix}/points3D.bin', colmap_utils.read_points3d_binary)
        self.images_registered = _safe_read(f'{prefix}/images.bin', colmap_utils.read_images_binary)

        # Build point cloud
        if self.points_raw is not None:
            pts, clr = [], []
            for v in self.points_raw:
                xyz, rgb = v.xyz, v.rgb
                pts.append(xyz)
                clr.append(rgb)
            self.pcd_pts = np.stack(pts, 0)
            self.pcd_clr = np.stack(clr, 0)
    
    @cached_property
    def summary(self):
        num_models = len(os.listdir(self.prefix))
        observes = [sum(v.point3D_ids > 0) for v in self.images_registered]

        track_lens, errors = [], []
        for v in self.points_raw:
            track_lens.append(len(v.image_ids))
            errors.append(v.error)

        r = {
            'reg_images': len(self.images_registered),
            'reproj_error': np.mean(errors),
            'track_length': np.mean(track_lens),
            'obs_per_img': np.mean(observes),
            'total_obs': np.sum(observes),
            'num_models': num_models,
        }
        return r
    
    def print_summary(self):
        """ Mimic the behavior of `colmap model_analyzer` """
        s = self.summary
        print(
            f'Num Models:\t{s["num_models"]}\n'
            f'Cameras:\t{len(self.cameras_raw)}\n'
            # f'Images:\t?'
            f'Registered Images:\t{s["reg_images"]}\n'
            f'Points:\t{len(self.points_raw)}\n'
            f'Observations:\t{s["total_obs"]}\n'
            f'Mean track length:\t{s["track_length"]:.6f}\n'
            f'Mean observations per image:\t{s["obs_per_img"]:.6f}\n'
            f'Mean reprojection error:\t{s["reproj_error"]:.6f}px'
        )
    
    @cached_property
    def database(self):

        def _make_df(fields, results):
            data = dict()
            for i, f in enumerate(fields):
                data[f] = []
            for res in results:
                for i, f in enumerate(fields):
                    data[f].append(res[i])
            return pd.DataFrame(data=data)
        
        with sqlite3.connect(self.database_path) as conn:
            cur = conn.cursor()
            
            cur.execute('SELECT * from cameras')
            cameras = _make_df(
                ['camera_id', 'model', 'width', 'height',
                'params', 'prior_focal_length'], results=cur.fetchall())

            cur.execute('SELECT * from descriptors')
            descriptors = _make_df(
                ['image_id', 'rows', 'cols', 'data'],
                results=cur.fetchall())
            
            cur.execute('SELECT * from images')
            images = _make_df(
                ['image_id', 'name', 'camera_id', 'prior_qw', 'prior_qx', 
                'prior_qy', 'prior_qz', 'prior_tx', 'prior_ty', 'prior_tz'],
                results=cur.fetchall())

            cur.execute('SELECT * from keypoints')
            keypoints = _make_df(
                ['image_id', 'rows', 'cols', 'data'],
                results=cur.fetchall())

            cur.execute('SELECT * from matches')
            matches = _make_df(
                ['pair_id', 'rows', 'cols', 'data'],
                results=cur.fetchall())

            cur.execute('SELECT * from two_view_geometries')
            two_view_geometries = _make_df(
                ['pair_id', 'rows', 'cols', 'data',
                'config', 'F', 'E', 'H'],
                #  'qvec', 'tvec'],
                results=cur.fetchall())
        
        return ColmapDB(**dict(
            cameras=cameras,
            descriptors=descriptors,
            images=images,
            keypoints=keypoints,
            matches=matches,
            two_view_geometries=two_view_geometries
            ))

    @property
    def pcd(self):
        return create_pcd_scene(
            points=self.pcd_pts,
            colors=self.pcd_clr, ret_pcd=True)
    
    @property
    def pcd_scene(self):
        return create_pcd_scene(
            points=self.pcd_pts,
            colors=self.pcd_clr, ret_pcd=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('proj_dir')
    return parser.parse_args()
    
if __name__ == '__main__':
    import subprocess
    args = parse_args()
    use_colmap = True
    if not use_colmap:
        proj = SparseProj(args.proj_dir)
        proj.print_summary()
    else:
        proc = subprocess.run(
            [
                'colmap', 'model_analyzer',
                '--path', f'{args.proj_dir}/sparse/0'
            ], 
            stdout=None, 
            check=True, text=True)