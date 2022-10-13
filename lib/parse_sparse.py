from typing import NamedTuple
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
    
    def __init__(self, proj_root):
        self.cameras_raw = colmap_utils.read_cameras_binary(f'{proj_root}/sparse/0/cameras.bin')
        self.points_raw = colmap_utils.read_points3d_binary(f'{proj_root}/sparse/0/points3D.bin')
        self.images_raw = colmap_utils.read_images_binary(f'{proj_root}/sparse/0/images.bin')

        # Build point cloud
        pts, clr = [], []
        for k, v in self.points_raw.items():
            xyz, rgb = v.xyz, v.rgb
            pts.append(xyz)
            clr.append(rgb)
        self.pcd_pts = np.stack(pts, 0)
        self.pcd_clr = np.stack(clr, 0)

        self.database_path = f'{proj_root}/database.db'
    
    @property
    def database(self):

        if hasattr(self, '_database') and self._database is not None:
            return self._database

        def _make_df(fields, results):
            data = dict()
            for i, f in enumerate(fields):
                data[f] = []
            for res in results:
                for i, f in enumerate(fields):
                    data[f].append(res[i])
            return pd.DataFrame(data=data)
        
        conn = sqlite3.connect(self.database_path)
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
             'config', 'F', 'E', 'H', 'qvec', 'tvec'],
            results=cur.fetchall())
        
        self._database = ColmapDB(**dict(
            cameras=cameras,
            descriptors=descriptors,
            images=images,
            keypoints=keypoints,
            matches=matches,
            two_view_geometries=two_view_geometries
            ))
        return self._database

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