import open3d as o3d
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    ply = o3d.io.read_point_cloud(args.model)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    t = - np.float32([0.04346319,1.05888072,2.09330869])
    rot = ply.get_rotation_matrix_from_xyz((-np.pi*15/180, 200*np.pi/180, 0))
    ply.translate(t)
    ply = ply.rotate(rot, center=(0, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(ply, reset_bounding_box=True)
    # vis.add_geometry(mesh_frame, reset_bounding_box=True)

    control = vis.get_view_control()
    control.set_front([1, 1, 1])
    control.set_lookat([0, 0, 0])
    control.set_up([0, 0, 1])
    control.set_zoom(0.13)

    vis.run()
    vis.destroy_window()