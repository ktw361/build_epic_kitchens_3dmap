from argparse import ArgumentParser
import json
import open3d as o3d
import numpy as np

import matplotlib.pyplot as plt

from lib.base_type import ColmapModel


""" Model will be read from 
<model_prefix><models_vid><model_suffix>/{cameras.bin, images.bin, points3D.bin}

"""

def generate_colormap(num_colors):
    colormap = plt.get_cmap("hsv", num_colors)
    color_dict = {}
    inds = np.arange(num_colors)
    np.random.seed(42)
    inds = np.random.permutation(inds)
    for i in range(num_colors):
        color_name = f"color_{i}"
        color_dict[color_name] = colormap(inds[i])[:3]
    return color_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--model_prefix', default='projects/')
    parser.add_argument('--model_suffix', default='')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--select', type=str, default=None)
    parser.add_argument('--line-length', type=float, default=10)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    infile = args.infile
    with open(infile, 'r') as fp:
        model_infos = json.load(fp)
    
    colors = generate_colormap(32)
    # colors = dict(
    #     red=[1, 0, 0],
    #     green=[0, 1, 0],
    #     blue=[0, 0, 1],
    #     yellow=[1, 1, 0],
    #     purple=[1, 0, 1],
    #     cyan=[0, 1, 1],
    #     white=[1, 1, 1],
    #     black=[0, 0, 0],
    # )
    alpha = args.alpha

    pcds = []
    line_sets = []

    for model_info, clr in zip(model_infos, colors.values()):
        if args.select is not None and model_info['model_vid'] != args.select:
            continue
        model_path = args.model_prefix + model_info['model_vid'] + args.model_suffix
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

        if 'line' in model_info:
            line = model_info['line']
            line = np.asarray(line).reshape(-1, 3)
            line = line * scale @ rot.T + transl
            vc = (line[0, :] + line[1, :]) / 2
            line_dir = line[1, :] - line[0, :]

            line_set = o3d.geometry.LineSet()
            line_len_half = args.line_length / 2
            lst = vc + line_len_half * line_dir
            led = vc - line_len_half * line_dir
            lines = [lst, led]
            line_set.points = o3d.utility.Vector3dVector(lines)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([np.asarray(clr)])
            line_sets.append(line_set)

    geoms = line_sets + pcds

    o3d.visualization.draw_geometries(geoms)
    # o3d.visualization.draw_geometries_with_editing([pcds])