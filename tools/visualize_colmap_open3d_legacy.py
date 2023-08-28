import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import json
from lib.base_type import ColmapModel
from tools.visualise_data_open3d import get_c2w, get_frustum

"""TODO
1. Frustum, on/off
2. Line (saved in json)
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--line')
    parser.add_argument('--show-mesh-frame', default=False)
    parser.add_argument('--specify-frame-name', default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model_path = args.model
    if model_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(model_path)
    else:
        mod = ColmapModel(args.model)
        pcd_np = [v.xyz for v in mod.points.values()]
        pcd_rgb = [v.rgb / 255 for v in mod.points.values()]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.colors = o3d.utility.Vector3dVector(pcd_rgb)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd, reset_bounding_box=True)
    if args.show_mesh_frame:
        vis.add_geometry(mesh_frame, reset_bounding_box=True)

    if args.specify_frame_name is not None:
        qvec, tvec = [
            (v.qvec, v.tvec) for k, v in mod.images.items() if v.name == args.specify_frame_name][0]
        img_data = [qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2]]
        c2w = get_c2w(img_data)
        frustum = get_frustum(c2w, sz=0.1, camera_height=256, camera_width=256)
        vis.add_geometry(frustum, reset_bounding_box=True)

    if args.line is not None:
        line_set = o3d.geometry.LineSet()
        with open(args.line, 'r') as f:
            line_points = np.asarray(json.load(f)).reshape(2, 3)
        vc = line_points.mean(axis=0)
        dir = line_points[1] - line_points[0]
        lst = vc + 2 * dir
        led = vc - 2 * dir
        lines = [lst, led]
        line_set.points = o3d.utility.Vector3dVector(lines)
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        vis.add_geometry(line_set, reset_bounding_box=True)


    control = vis.get_view_control()
    control.set_front([1, 1, 1])
    control.set_lookat([0, 0, 0])
    control.set_up([0, 0, 1])
    control.set_zoom(1.0)

    vis.run()
    vis.destroy_window()
