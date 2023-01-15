import argparse
import os
import os.path as osp
import sqlite3
import numpy as np
import re
from colmap_converter.read_write_model import (
    read_model, write_model, Image, Point3D
)

"""
python tools/reindex_database.py proj_path

image_path must contains Pxx_yyy/zzz...z.jpg

According to video_info of Epic-55 and Epic-100,
the max number of frames are less than (222482, 221560) respectively, which is no more than 6 digits;
The INTEGER type in database has maximal value 2147483467 ~ 2e9,
hence we can reindex the image_id = sub(3-digit) + frame_id(6-digit) < 1e9 < 2e9.
We drop the two digit pid because frames must come from the same kitchen(same pid).
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('proj_path')
    return parser.parse_args()

def reindex_project(proj_path):
    """ change image_id to video_id + image_id,
    so if the there are common images in two models, they will have the same image_id
    """
    db_path = osp.join(proj_path, 'database.db')
    reindex_database(db_path)
    for model in os.listdir(osp.join(proj_path, 'sparse')):
        sparse_path = osp.join(proj_path, 'sparse', model)
        reindex_sparse(sparse_path)


def image_name_to_id(image_name):
    # input P23_101, output (23, 101)
    _, sub = re.search(r'P(\d{2,})_(\d{2,3})', image_name).groups()
    frame_id = re.search(r'(\d{10})', image_name).groups()[0]
    # convert (sub, frame_id) to sub*10^6 + frame_id
    ret = int(sub) * 10**6 + int(frame_id)
    return ret


def reindex_database(db_path):
    # TODO: also reindex keypoints, matches, two_view_geometries, descriptors
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute('SELECT * from images')
        results = cur.fetchall()
        for res in results:
            image_id = res[0]
            image_path = res[1]
            # print(image_id, image_path)
            new_image_id = image_name_to_id(image_path)
            # print(new_image_id)
            cur.execute(
                f'update images set image_id = {new_image_id} where image_id = {image_id};')

        # cur.execute('SELECT * from images')
        # results = cur.fetchall()
        # for res in results:
        #     print(res)
    
    # write back to database
    conn.commit()


def reindex_sparse(model_path):
    """ model_path: path to sparse/0 """
    cameras, images, points3D = read_model(model_path)
    id_map = {}  # mapping from old image_id to new image_id
    new_images = {}
    for old_id, img in images.items():
        name = img.name
        new_image_id = image_name_to_id(name)
        new_image = Image(
            id=new_image_id, qvec=img.qvec, tvec=img.tvec, 
            camera_id=img.camera_id, name=name, xys=img.xys, 
            point3D_ids=img.point3D_ids)
        new_images[new_image_id] = new_image
        assert old_id not in id_map
        id_map[old_id] = new_image_id

    for pt_id, pt in points3D.items():
        image_ids = np.array([id_map[old_id] for old_id in pt.image_ids])
        points3D[pt_id] = Point3D(
            id=pt_id, xyz=pt.xyz, rgb=pt.rgb, error=pt.error,
            point2D_idxs=pt.point2D_idxs, image_ids=image_ids)
    
    write_model(cameras, new_images, points3D, path=model_path, ext='.bin')


if __name__ == '__main__':
    reindex_project(parse_args().proj_path)
    