import open3d as o3d
import numpy as np
from argparse import ArgumentParser
from manual_merge.ann_model import AnnotatedModel
from manual_merge.functions import compute_sim3_transform

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model-1')
    parser.add_argument('--model-2')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    mod1 = AnnotatedModel(args.model_1)
    mod2 = AnnotatedModel(args.model_2)

    rot, transl, scale, mat, err = compute_sim3_transform(
        mod1.ordered_landmarks, mod2.ordered_landmarks)

    print('mat: ', mat)
    print('scale: ', scale)
    print('err: ', err.reshape(-1, 1))

    pcd1_np = [v.xyz for v in mod1.points.values()]
    pcd1_rgb = [v.rgb / 255 for v in mod1.points.values()]
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_np)
    pcd1.colors = o3d.utility.Vector3dVector(pcd1_rgb)

    pcd2_np = [v.xyz * scale @ rot.T + transl for v in mod2.points.values()]
    pcd2_rgb = [v.rgb / 255 for v in mod2.points.values()]
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_np)
    pcd2.colors = o3d.utility.Vector3dVector(pcd2_rgb)

    o3d.visualization.draw_geometries([pcd1, pcd2])