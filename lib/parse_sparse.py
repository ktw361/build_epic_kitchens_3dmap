from typing import NamedTuple
from functools import cached_property
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

        as_list = lambda x: list(dict(x).values())
        self.cameras_raw = as_list(colmap_utils.read_cameras_binary(f'{prefix}/cameras.bin'))
        self.points_raw  = as_list(colmap_utils.read_points3d_binary(f'{prefix}/points3D.bin'))
        self.images_registered = as_list(colmap_utils.read_images_binary(f'{prefix}/images.bin'))

        # Build point cloud
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
        num_observes = np.mean(
            [sum(v.point3D_ids > 0) for v in self.images_registered])

        track_lens, errors = [], []
        for v in self.points_raw:
            track_lens.append(len(v.image_ids))
            errors.append(v.error)

        r = {
            'reg_images': len(self.images_registered),
            'reproj_error': np.mean(errors),
            'track_length': np.mean(track_lens),
            'obs_per_img': num_observes,
            'num_models': num_models,
        }
        return r
    
    def print_summary(self):
        for k, v in self.summary.items():
            print(f'{k}\t=\t{v:.3f}')
    
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

    
if __name__ == '__main__':
    pass