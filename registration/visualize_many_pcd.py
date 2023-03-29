from argparse import ArgumentParser
import json
import open3d as o3d
import numpy as np
from lib.base_type import ColmapModel

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--alpha', type=float, default=0.5)
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
    alpha = args.alpha
    pcds = []

    for model_info, clr in zip(model_infos, colors.values()):
        model_path = model_info['model_path']
        model = ColmapModel(model_path)
        rot = np.asarray(model_info['rot']).reshape(3, 3)
        transl = np.asarray(model_info['transl'])
        scale = model_info['scale']

        pcd_np = [
            v.xyz * scale @ rot.T + transl
            for v in model.points.values()]
        pcd_rgb = [
            alpha * np.float32(v.rgb / 255) + (1-alpha) * np.float32(clr)
            for v in model.points.values()]
        # pcd_rgb = [clr for v in model.points.values()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)