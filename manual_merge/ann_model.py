import os.path as osp
import json
from collections import OrderedDict
import numpy as np

from lib.base_type import ColmapModel


""" 
Prototype

example CounterTop
d---------c
|  TableA |
a---------b

Each landmark is a named point of shape (1, 3)
    - support vicinity points (N, 3) ?
    e.g. 'TableA-a': [0, 0, 1]

We can form 
    - a line
        'A-ab': ['A-a', 'A-b']
    - a parallelogram
        'A': ['A-a', 'A-b', 'A-c', 'A-d']

We can have two annotation type: a video dependent and a kitchen structure.
video file P01_01.json
{
    'A-a': [0, 0, 0]
}

kitchen structured (shared)
{
    'A': ['A-a', ...]
}

"""


class AnnotatedModel(ColmapModel):
    """
    Each AnntatedModel describes the colmap output triple + annotated landmarks
        <camera, images, points, landmarks: dict = None> 
    """
    
    def __init__(self, 
                 model_dir: str,
                 landmark_file=None):
        """
        """
        super().__init__(model_dir)
        if landmark_file is None:
            landmark_file = osp.join(model_dir, "landmarks.json")

        if osp.exists(landmark_file):
            with open(landmark_file) as fp:
                d = json.load(fp)
            self.landmark_dict = OrderedDict(sorted(d.items()))
        else:
            self.landmark_dict = None
        
    @property
    def ordered_landmarks(self) -> np.ndarray:
        """ returns: (L, 3) """
        if self.landmark_dict is None:
            return None

        # TODO: process redundant landmarks?
        points = [
            np.asarray(v)[0]
            for v in self.landmark_dict.values()
        ]
        return np.asarray(points)
