from pathlib import Path
from functools import cached_property
from argparse import ArgumentParser
import glob
from PIL import Image
from omegaconf import OmegaConf
import os
import numpy as np
from colmap_converter import colmap_utils
import cv2
from lib.database_utils import parse_database
from libzhifan.geometry import create_pcd_scene
from libzhifan import epylab


class SparseProj:
    
    def __init__(self, proj_root, model_id=0, prefix=None):
        proj_root = Path(proj_root)
        if prefix is None:
            prefix = f'sparse/{model_id}'
        prefix = f'{proj_root}/{prefix}'
            
        self.database_path = f'{proj_root}/database.db'
        self.prefix = prefix

        def _safe_read(path, func):
            if not os.path.exists(path):
                print(f'{path} not exists.')
                return None
            as_list = lambda x: list(dict(x).values())
            return as_list(func(path))
        self.cameras_raw = _safe_read(f'{prefix}/cameras.bin', colmap_utils.read_cameras_binary)
        self.points_raw  = _safe_read(f'{prefix}/points3D.bin', colmap_utils.read_points3d_binary)
        images_registered = _safe_read(f'{prefix}/images.bin', colmap_utils.read_images_binary)
        if images_registered is not None:
            self.images_registered = sorted(images_registered, key = lambda x: x.name)

        if os.path.exists(proj_root/'.hydra/config.yaml'):
            cfg = OmegaConf.load(proj_root/'.hydra/config.yaml')
            self.image_path = proj_root/'images'
            self.is_nomask = False
            self.is_simplemask = False
            if cfg.is_nomask:
                self.is_nomask = True
            elif cfg.is_simplemask:
                self.is_simplemask = True
                self.camera_mask_path = glob.glob(str(proj_root/'*.png'))[0]
            else:
                self.mask_path = proj_root/'masks'

        # Build point cloud
        if self.points_raw is not None:
            pts, clr = [], []
            for v in self.points_raw:
                xyz, rgb = v.xyz, v.rgb
                pts.append(xyz)
                clr.append(rgb)
            self.pcd_pts = np.stack(pts, 0)
            self.pcd_clr = np.stack(clr, 0)
    
    @cached_property
    def summary(self):
        num_models = len(os.listdir(self.prefix))
        observes = [sum(v.point3D_ids > 0) for v in self.images_registered]

        track_lens, errors = [], []
        for v in self.points_raw:
            track_lens.append(len(v.image_ids))
            errors.append(v.error)

        r = {
            'reg_images': len(self.images_registered),
            'reproj_error': np.mean(errors),
            'track_length': np.mean(track_lens),
            'obs_per_img': np.mean(observes),
            'total_obs': np.sum(observes),
            'num_models': num_models,
        }
        return r
    
    def print_summary(self):
        """ Mimic the behavior of `colmap model_analyzer` """
        s = self.summary
        print(
            f'Num Models:\t{s["num_models"]}\n'
            f'Cameras:\t{len(self.cameras_raw)}\n'
            # f'Images:\t?'
            f'Registered Images:\t{s["reg_images"]}\n'
            f'Points:\t{len(self.points_raw)}\n'
            f'Observations:\t{s["total_obs"]}\n'
            f'Mean track length:\t{s["track_length"]:.6f}\n'
            f'Mean observations per image:\t{s["obs_per_img"]:.6f}\n'
            f'Mean reprojection error:\t{s["reproj_error"]:.6f}px'
        )
    
    @cached_property
    def database(self):
        return parse_database(self.database_path)
        
    def _show_img_kps(self, img_path, xys, show_origin, mask_name: str = None):
        """
        Args:
            img_path: path to a single image
            xys: [N, 2] keypoints
            show_origin: if False, black-out mask region, 
                this requires mask_name to be valid
        """
        img = np.asarray(Image.open(img_path))
        if not show_origin:
            if self.is_nomask:
                pass
            elif self.is_simplemask:
                mask = np.asarray(Image.open(f'{self.camera_mask_path}'))
                img[mask[..., :3] == 0] = 0
            else:
                mask = np.asarray(Image.open(f'{self.mask_path}/{mask_name}'))
                img[mask[..., :3] == 0] = 0
        for x, y in xys:
            img = cv2.circle(
                img, (int(x), int(y)), radius=1, color=(255, 0, 0), thickness=4)
        return img

    def show_keypoints(self, start_idx, num_imgs, show_origin=False):
        """ Show keypoints in the sparse/0 dir """
        img_metas = self.images_registered

        def single_show_keypoint(idx) -> np.ndarray:
            img_meta = img_metas[idx]
            img_name = img_meta.name
            mask_name = img_name + '.png'
            img_path = os.path.join(self.image_path, img_name)
            return self._show_img_kps(img_path=img_path, xys=img_meta.xys, 
                                      show_origin=show_origin, mask_name=mask_name)

        for idx in range(start_idx, start_idx + num_imgs + 1):
            epylab.figure()
            img_meta = img_metas[idx]
            epylab.title(img_meta.name)
            epylab.axis('off')
            img = single_show_keypoint(idx)
            epylab.imshow(img)
        epylab.close()

    def show_database_keypoints(self, 
                                img_dir,
                                image_id: int = None, 
                                image_name: str = None,
                                num_cols=6) -> np.ndarray:
        """ Show keypoints stored in database.db, i.e. output of feature_extractor.
         
        6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)
        """
        assert num_cols == 6
        df = self.database.keypoints
        if image_id is None and image_name is not None:
            image_id = self.image_name2id(image_name)
        if image_name is None:
            image_name = self.image_id2name(image_id)
        img_path = os.path.join(img_dir, image_name)
        entry = df[df.image_id == image_id].iloc[0]
        array = np.frombuffer(entry.data, dtype=np.float32).reshape(-1, num_cols)
        xys = array[:, :2]
        return self._show_img_kps(img_path=img_path, xys=xys,
                                  show_origin=True, mask_name=None)

    def image_id2name(self, i: int) -> str:
        df = self.database.images
        return df[df.image_id == i].name.iloc[0]
    
    def image_name2id(self, name: str) -> int:
        df = self.database.images
        return df[df.name == name].image_id.iloc[0]
    
    @property
    def pcd(self):
        return create_pcd_scene(
            points=self.pcd_pts,
            colors=self.pcd_clr, ret_pcd=True)
    
    @property
    def pcd_scene(self):
        return create_pcd_scene(
            points=self.pcd_pts,
            colors=self.pcd_clr, ret_pcd=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('proj_dir')
    return parser.parse_args()
    
if __name__ == '__main__':
    import subprocess
    args = parse_args()
    use_colmap = True
    if not use_colmap:
        proj = SparseProj(args.proj_dir)
        proj.print_summary()
    else:
        proc = subprocess.run(
            [
                'colmap', 'model_analyzer',
                '--path', f'{args.proj_dir}/sparse/0'
            ], 
            stdout=None, 
            check=True, text=True)