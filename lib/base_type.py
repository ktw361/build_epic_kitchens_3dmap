from typing import List
from functools import cached_property
from colmap_converter import colmap_utils


class ColmapModel:

    """
    NOTE: this class shares commons codes with line_check.LineChecker,
        reuse these codes?
    """
    def __init__(self, model_dir: str):

        def _as_list(path, func):
            return func(path)

        cameras = _as_list(
            f'{model_dir}/cameras.bin', colmap_utils.read_cameras_binary)
        assert len(cameras) == 1, "Only support single camera"
        self.camera = cameras[1]
        self.points = _as_list(
            f'{model_dir}/points3D.bin', colmap_utils.read_points3d_binary)
        self.images = _as_list(
            f'{model_dir}/images.bin', colmap_utils.read_images_binary)
    
    def __repr__(self) -> str:
        return f'{self.num_images} images - {self.num_points} points'

    @property
    def example_data(self):
        ki = list(self.images.keys())[0]
        img = self.images[ki]
        kp = list(self.points.keys())[0]
        point = self.points[kp]
        return img, point
    
    @cached_property
    def ordered_image_ids(self):
        return sorted(self.images.keys(), key=lambda x: self.images[x].name)
    
    @property
    def num_points(self):
        return len(self.points)
    
    @property
    def num_images(self):
        return len(self.images)
    
    @property
    def ordered_images(self) -> List[colmap_utils.BaseImage]:
        return [self.images[i] for i in self.ordered_image_ids]
    
    def get_image_by_id(self, image_id: int):
        return self.images[image_id]