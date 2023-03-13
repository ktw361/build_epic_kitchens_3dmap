# TODO

- [ ] Activate PCD


# Open3d

## 0.17.0 (Headless)
Need to compile with osmesa
Can' control camera
No OffscreenRender


## 0.16.0 (Offscreen) [Current Use]
pip install open3d==0.16.0 (need to apt install osmesa also?)

Need to decide best FOV visually in Open3D gui, but Tuning FOV is fun!

### Template
```python
render = rendering.OffscreenRenderer(640, 480)

render.setup_camera(60.0, 
    lookAt=[0, 0, 0], 
    location=[5, 10, 3], 
    up=[0, 0, 1])
render.scene.show_axes(True)

yellow = rendering.MaterialRecord()
yellow.base_color = [1.0, 0.75, 0.0, 1.0]
yellow.shader = "defaultLit"
render.scene.add_geometry('pcd', pcd, yellow)
render.scene.remove_geometry('box')

render.clear_geomtry()
```
Scene methods: http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Open3DScene.html?highlight=open3d%20scene#open3d.visualization.rendering.Open3DScene.scene

### Scratch
from scipy.spatial.transform import rotation
scene_rot = rotation.Rotation.from_euler('xyz', [180, 10, 150], degrees=True).as_matrix()
scene_transl = -1 * np.float32([0.89424676, 0.09415632, 3.33418347])  # pcd.get_center()