from functools import cached_property
import re
import os
from lib.base_type import ColmapModel
from registration.functions import compute_reproj_error


class EpicKitchensRegistrator:
    """
    In Mixed model, images are named Pxx_yy / Pxx_yyy, thus we know which video they are from
    """

    def __init__(self, 
                 mixed_model: ColmapModel):
        self.mixed_model = mixed_model
    
    @cached_property
    def registered_frames(self) -> dict:
        """
        In mixed model, images are already registered inter-video.
        This will give a dict of {video_name: [frame_name]}

        Returns:
            dict: {video_name: [frame_name]}
                -frame_name: frame_%010d.jpg
        """
        reg_frames = dict()
        for v in self.mixed_model.images.values():
            n = v.name
            vid = re.search('P\d{2,}_\d{2,3}', n)[0]
            frame_name = re.search('frame_\d{10}.jpg', n)[0]
            reg_frames.setdefault(vid, set()).add(frame_name)
        return reg_frames
    
    @property
    def registered_stats(self):
        return {k: len(v) for k, v in self.registered_frames.items()}

    def get_common_frames(self, single_model: ColmapModel, vid: str) -> set:
        """
        Returns:
            common_frames: set, frame_%010d.jpg
        """
        single_frames = [
            re.search('frame_\d{10}.jpg', v.name)[0]
            for v in single_model.images.values()]
        single_frames = set(single_frames)
        ref_frames = self.registered_frames[vid]
        common_frames = ref_frames.intersection(single_frames)
        return common_frames


    def register_to_single(self, single_model: ColmapModel, vid: str):
        """
        Args:
            single_model: ColmapModel
            vid: str, Pxx_yyy
        """
        common_frames = self.get_common_frames(single_model, vid)

        for frame_name in common_frames:
            # Compute: if I use this frame, how large the reproj error will be?
            
            # compute_reproj_error
            
            pass