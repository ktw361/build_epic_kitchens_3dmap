from argparse import ArgumentParser
import json
import open3d as o3d
import numpy as np
from lib.base_type import ColmapModel
# from registration.functions import compute_relative_pose
# from manual_merge.functions import compute_sim3_transform

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    infile = args.infile
    with open(infile, 'r') as fp:
        model_infos = json.load(fp)
    
    colors = dict(
        red=[1, 0, 0],
        green=[0, 1, 0],
        blue=[0, 0, 1],
        yellow=[1, 1, 0],
        purple=[1, 0, 1],
        cyan=[0, 1, 1],
        white=[1, 1, 1],
        black=[0, 0, 0],
    )
    pcds = []

    for model_info, clr in zip(model_infos, colors.values()):
        model_path = model_info['model_path']
        model = ColmapModel(model_path)
        m2a = model_info['model_to_common']

        pcd_np = [v.xyz for v in model.points.values()]
        alpha = 0.9
        pcd_rgb = [
            alpha * np.float32(v.rgb / 255) + (1-alpha) * np.float32(clr)
            for v in model.points.values()]
        # pcd_rgb = [clr for v in model.points.values()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
        pcds.append(pcd)

    # rot, transl, scale, mat, err = compute_sim3_transform(
    #     mod1.ordered_landmarks, mod2.ordered_landmarks)

    # print('mat: ', mat)
    # print('scale: ', scale)
    # print('err: ', err.reshape(-1, 1))

    # pcd1_np = [v.xyz for v in mod1.points.values()]
    # pcd1_rgb = [v.rgb / 255 for v in mod1.points.values()]
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(pcd1_np)
    # pcd1.colors = o3d.utility.Vector3dVector(pcd1_rgb)

    # pcd2_np = [v.xyz * scale @ rot.T + transl for v in mod2.points.values()]
    # pcd2_rgb = [v.rgb / 255 for v in mod2.points.values()]
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(pcd2_np)
    # pcd2.colors = o3d.utility.Vector3dVector(pcd2_rgb)

    o3d.visualization.draw_geometries(pcds)