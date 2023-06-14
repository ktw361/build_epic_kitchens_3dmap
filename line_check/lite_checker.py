from collections import namedtuple
from functools import cached_property
from typing import List, Union
import tqdm
import json, os, re, shutil
import os.path as osp
import numpy as np
import cv2
from PIL import Image
from moviepy import editor
from lib.base_type import ColmapModel, JsonColmapModel

from line_check.line import Line
from line_check.functions import project_line_image


class MultiLineChecker:
    """ A lite version of LineChecker that only visualise the line(s)
    without computing the error. """

    COLORS = dict(
        yellow=(255, 255, 0),
        white=(255, 255, 255),
        red=(255, 0, 0),
        green=(0, 255, 0),
        blue=(0, 0, 255),
    )
    
    def __init__(self,
                 model: Union[ColmapModel, JsonColmapModel],
                 anno_points_list: List[np.ndarray],
                 line_colors: List[str],
                 frames_root: str = None):

        if isinstance(model, ColmapModel):
            self.camera = model.camera
            self.images = model.images
        elif isinstance(model, JsonColmapModel):
            camera = model.camera
            images = model.images
            CustomCamera = namedtuple('CustomCamera', ['width', 'height', 'params'])
            CustomImage = namedtuple('CustomImage', ['qvec', 'tvec', 'name'])

            self.camera = CustomCamera(camera['width'], camera['height'], camera['params'])
            self.images = {
                i: CustomImage(np.asarray(image[:4]), np.asarray(image[4:7]), image[7])
                for i, image in enumerate(images)}

        self.lines = []
        for anno_points in anno_points_list:
            anno_points = np.asarray(anno_points).reshape(-1, 3)
            line = Line(anno_points)
            self.lines.append(line)

        self.line_colors = [self.COLORS[c] for c in line_colors]
        self.frames_root = frames_root if frames_root is not None else ""
    
    @cached_property
    def ordered_image_ids(self):
        return sorted(self.images.keys(), key=lambda x: self.images[x].name)
    
    @property
    def num_images(self):
        return len(self.images)
    
    @property
    def ordered_images(self):
        return [self.images[i] for i in self.ordered_image_ids]
    
    def get_image_by_id(self, image_id: int):
        return self.images[image_id]

    def visualize_compare(self, 
                          image_id: int, 
                          debug=False):
        """
        Red color: (prediction)
            Annotated Line projected to the image using Predicted camera pose.
        
        Green color: (GT)
            Fitted 2d line using points retrieved from the 3d line.
            2d keypoints are fixed, hence the 2d line is truth.
        
        Args:
            idx: int, camera pose index 
        """
        image = self.images[image_id]
        img_name = image.name
        img_path = osp.join(self.frames_root, img_name)
        img = np.asarray(Image.open(img_path))

        radius = 0.0
        for line, color in zip(self.lines, self.line_colors):
            line_2d, _, _ = project_line_image(
                line, radius, self.images[image_id], self.camera,
                debug=debug)
            if line_2d is None:
                continue
            img = cv2.line(
                img, np.int32(line_2d[0]), np.int32(line_2d[1]), 
                color=color, thickness=2, lineType=cv2.LINE_AA)
        return img
    
    def write_mp4(self, radius: float, out_name: str, fps=10, out_base='outputs/line_check/', delete_images=True):
        """ Write mp4 file that project the line on the image 

        Args:
            out_name: e.g. P01_01
                - images will be written to <out_base>/<out_name>/frame_%010d.jpg
                - mp4 will be written to <out_base>/<out_name>.mp4
        """
        out_dir = os.path.join(out_base, out_name)
        os.makedirs(out_dir, exist_ok=True)
        fmt = os.path.join(out_dir, '{}')

        for img_id in tqdm.tqdm(self.ordered_image_ids):
            name = self.images[img_id].name
            img = self.visualize_compare(img_id) # , display=display)
            frame_number = re.search('\d{10,}', 
                                    self.get_image_by_id(img_id).name)[0]
            cv2.putText(img, frame_number, 
                        (self.camera.width//4, self.camera.height * 31 // 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            frame = os.path.basename(name)
            Image.fromarray(img).save(fmt.format(frame))

        clip = editor.ImageSequenceClip(sequence=out_dir, fps=fps)
        video_file = os.path.join(out_base, f'{out_name}-fps{fps}.mp4')
        clip.write_videofile(video_file)
        if delete_images:
            shutil.rmtree(out_dir)


class JsonMultiLineChecker(MultiLineChecker):
    """ A lite version of LineChecker that only visualise the line(s)
    without computing the error. """
    
    def __init__(self,
                 camera: dict,
                 images: List[List],
                 anno_points_list: List[np.ndarray],
                 line_colors: List[str],
                 frames_root: str = None):
        """
        Args:
            camera: dict, camera info
            images: list of image
                [qw, qx, qy, qz, tx, ty, tz, name]
        """
        CustomCamera = namedtuple('CustomCamera', ['width', 'height', 'params'])
        CustomImage = namedtuple('CustomImage', ['qvec', 'tvec', 'name'])

        self.camera = CustomCamera(camera['width'], camera['height'], camera['params'])
        self.images = {
            i: CustomImage(np.asarray(image[:4]), np.asarray(image[4:7]), image[7])
            for i, image in enumerate(images)}

        self.lines = []
        for anno_points in anno_points_list:
            anno_points = np.asarray(anno_points).reshape(-1, 3)
            line = Line(anno_points)
            self.lines.append(line)

        self.line_colors = [self.COLORS[c] for c in line_colors]
        self.frames_root = frames_root if frames_root is not None else ""