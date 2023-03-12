import open3d as o3d
import numpy as np
from argparse import ArgumentParser
from manual_merge.ann_model import AnnotatedModel

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    mod = AnnotatedModel(args.model)


    pcd_np = [v.xyz for v in mod.points.values()]
    pcd_rgb = [v.rgb / 255 for v in mod.points.values()]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)

    o3d.visualization.draw_geometries([pcd])