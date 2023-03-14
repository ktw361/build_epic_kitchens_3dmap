import open3d as o3d
import numpy as np
from argparse import ArgumentParser

from lib.base_type import ColmapModel
from hovering.helper import (
    get_frustum, get_o3d_pcd
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    mod = ColmapModel(args.model)
    pcd = get_o3d_pcd(mod)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    cimg = mod.get_image_by_id(885)
    frustum = get_frustum(
        sz=1, line_radius=0.05, colmap_image=cimg,
        camera_height=mod.camera.height, camera_width=mod.camera.width)

    t = - np.float32([0.04346319,1.05888072,2.09330869])
    pcd = pcd.translate(t)
    rot = pcd.get_rotation_matrix_from_xyz(
        [-np.pi*15/180, 180*np.pi/180, -30 * np.pi / 180])
    pcd = pcd.rotate(rot, center=(0, 0, 0))
    # pcd = pcd.translate([10, 7, 9])

    frustum = frustum.translate(t).rotate(rot, center=(0, 0, 0)) # .translate([10, 7, 9])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd, reset_bounding_box=True)
    vis.add_geometry(frustum, reset_bounding_box=True)
    vis.add_geometry(mesh_frame, reset_bounding_box=True)

    control = vis.get_view_control()
    control.set_front([10, 10, 10])
    control.set_lookat([0, 0, 0])
    control.set_up([0, 0, 1])
    control.set_zoom(0.13)

    vis.run()
    vis.destroy_window()