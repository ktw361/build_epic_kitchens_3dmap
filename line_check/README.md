# STEPS

1. In COLMAP-GUI, click key point to copy their 3D positions. (1~5mins)
    - modify the `anno_points` in `run_checker.py` and `visualize_line.py` TODO: Seperate out the `anno_points` as arguments.
    - Select more than 2 points so that least-square fitting can estimate line more robustly.

2. On a local machine, run `python line_check/visualize_line.py --model-dir <path-to-sparse/0>` to eye-check the LS fitted line is what you want.
    - Need to install open3d for this step.

3. On remote/local machine, run `python line_check/run_checker.py <path-to-sparse/0>`
    - Install `pytorch3d` for quaternion conversion.

## TODO / Hyperparameters

- [ ] 2D line angle as error? 
- radius is a hyperparameter


# Note

1. For Open3D WebRTC visualization, `import open3d as o3d` must be put at the very beginning?