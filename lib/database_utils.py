from typing import NamedTuple
import sqlite3
import pandas as pd

class ColmapDB(NamedTuple):
    cameras: pd.DataFrame
    descriptors: pd.DataFrame
    images: pd.DataFrame
    keypoints: pd.DataFrame
    matches: pd.DataFrame
    two_view_geometries: pd.DataFrame


def parse_database(database_path) -> ColmapDB:
    
    def _make_df(fields, results):
        data = dict()
        for i, f in enumerate(fields):
            data[f] = []
        for res in results:
            for i, f in enumerate(fields):
                data[f].append(res[i])
        return pd.DataFrame(data=data)
    
    with sqlite3.connect(database_path) as conn:
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