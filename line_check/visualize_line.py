import open3d as o3d
import numpy as np
from argparse import ArgumentParser
from line_check.checker import LineChecker

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model-dir')
    return parser.parse_args()

if __name__ == "__main__":
    # o3d.visualization.webrtc_server.enable_webrtc()
    args = parse_args()
    anno_points = [
-5.38907, 0.687732, 4.77714,
4.70254, -2.36873, 2.34974,
0.700261, -1.09484, 2.75475,
            # 4.63313, -0.2672, 2.55641,
            # -5.22596, 0.352575, 3.04684,
            # 0.675789, -0.0019428, 2.77022
            ]
    anno_points = np.asarray(anno_points).reshape(-1, 3)
    checker = LineChecker(args.model_dir,
                          anno_points=anno_points)

    #_ = checker.aggregate(radius=radius, return_dict=True)

    line_set = o3d.geometry.LineSet()
    lst = checker.line.vc + 100 * checker.line.dir
    led = checker.line.vc - 100 * checker.line.dir
    lines = [lst, led]
    line_set.points = o3d.utility.Vector3dVector(lines)
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    # create a point cloud
    points_np = [v.xyz for v in checker.points.values()]
    points_rgb = [v.rgb / 255 for v in checker.points.values()]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(points_rgb)

    o3d.visualization.draw_geometries([line_set, pcd])

    # o3d.visualization.draw([line_set, pcd])