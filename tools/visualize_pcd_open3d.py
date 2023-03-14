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
    print("Oriented Bounding Box: ", pcd.get_oriented_bounding_box())

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    t = - np.float32([0.04346319,1.05888072,2.09330869])
    t = pcd.get_min_bound()
    rot = pcd.get_rotation_matrix_from_xyz((-np.pi*15/180, 200*np.pi/180, 0))
    pcd.translate(t)
    pcd = pcd.rotate(rot, center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd, reset_bounding_box=True)
    # vis.add_geometry(mesh_frame, reset_bounding_box=True)

    control = vis.get_view_control()
    control.set_front([1, 1, 1])
    control.set_lookat([0, 0, 0])
    control.set_up([0, 0, 1])
    control.set_zoom(0.13)

    vis.run()
    vis.destroy_window()