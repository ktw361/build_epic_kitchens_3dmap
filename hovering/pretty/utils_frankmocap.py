from typing import List
import sys
sys.path.insert(0, '/home/skynet/Zhifan/ihoi/')
sys.path.insert(0, '/home/skynet/Zhifan/ihoi/externals/frankmocap')
sys.path.append('/home/skynet/Zhifan/ihoi/externals/frankmocap/detectors/body_pose_estimator/')

import re
import tqdm
import numpy as np
import torch
from PIL import Image

from datasets.epic_lib import epichoa
import pytorch3d.transforms.rotation_conversions as rot_cvt
from homan.homan_ManoModel import HomanManoModel
from nnutils.handmocap import (
    recover_pca_pose, get_hand_faces, collate_mocap_hand,
    get_handmocap_predictor, get_hand_wrapper
)
from nnutils import geom_utils
from lib.common_functions import colmap_image_c2w
from pytorch3d.transforms import rotation_6d_to_matrix

from libzhifan import odlib
from libzhifan.geometry import SimpleMesh, visualize_mesh, projection, CameraManager, BatchCameraManager


def get_cam_manager(colmap_camera):
    # fx, fy, cx, cy, _, _, _, _ = colmap_camera.params
    # img_w, img_h = colmap_camera.width, colmap_camera.height
    # return CameraManager(fx, fy, cx, fy, img_h, img_w, in_ndc=False)
    fx, fy, cx, cy, _, _, _, _ = colmap_camera['params']
    img_w, img_h = colmap_camera['width'], colmap_camera['height']
    return CameraManager(fx, fy, cx, fy, img_h, img_w, in_ndc=False)


def compute_hand_transform(rot_axisang,
                           pred_hand_pose,
                           pred_camera,
                           side: str,
                           hand_cam: BatchCameraManager):
    """
    Args:
        rot_axisang: (B, 3)
        pred_hand_pose: (B, 45)
        pred_camera: (B, 3)
            Used for translate hand_mesh to convert hand_mesh
            so that result in a weak perspective camera.
        hand_wrapper: ManoPthWrapper

    Returns:
        rotation6d: (B, 3, 3), will apply to row-vecs
        translation: (B, 1, 3)
    """
    rotation = rot_cvt.axis_angle_to_matrix(rot_axisang)  # (1, 3) - > (1, 3, 3), col-vec
    rot_homo = geom_utils.rt_to_homo(rotation)
    glb_rot = geom_utils.matrix_to_se3(rot_homo)  # (1, 4, 4) -> (1, 12)
    _, joints = get_hand_wrapper(side)(
        glb_rot,
        pred_hand_pose, return_mesh=True)
    s, tx, ty = torch.split(pred_camera, [1, 1, 1], dim=1)
    device = tx.device

    fx, fy, cx, cy, _, _ = hand_cam.unpack()
    fx, fy, cx, cy = map(
        lambda x: torch.as_tensor(x, device=device).view_as(s),
        (fx, fy, cx, cy))
    f = (fx + fy) / 2  # How to enforce fx=fy?

    FRANKMOCAP_INPUT_SIZE = 224
    sw  = s * FRANKMOCAP_INPUT_SIZE /2  # s*w/2
    # print(s, tx, ty, sw)
    tx = tx + 1/s - cx / sw
    ty = ty + 1/s - cy / sw
    tz = f / sw
    translation = torch.cat([tx, ty, tz], dim=1)
    translation = translation - joints[:, 5]
    rotation6d = rot_cvt.matrix_to_rotation_6d(rotation)
    return rotation6d, translation[:, None]


class HandGetter:
    """ up
    P28_101_magics = dict(
        left=dict(vscale=2.5, tscale=2.5, trans=torch.tensor([0, 0.23, 0], device='cuda', dtype=torch.float32)),
        right=dict(vscale=2.5, tscale=2.5, trans=torch.tensor([0, 0.20, 0], device='cuda', dtype=torch.float32))
    )
    """
    def __init__(self, vid):
        self.vid = vid
        self.df = epichoa.load_video_hoa(vid, '/home/skynet/Zhifan/datasets/epic/hoa')
        self.epic_rgb_root = '/home/skynet/Zhifan/data/epic_rgb_frames'
        self.frames_root = f'{self.epic_rgb_root}/{vid[:3]}/{vid}/'
        self.hand_predictor = get_handmocap_predictor()

    def get_hand_boxes(self, frame, as_dict=True, pad=0.2):
        """ xywh """
        EPIC_HOA_SIZE = (1920, 1080)
        # VISOR_SIZE = (854, 480)
        EPIC_SIZE = (456, 256)

        def row2xywh(row):
            wid = row.right - row.left
            hei = row.bottom - row.top
            l = row.left - pad * wid
            r = row.right + pad * wid
            t = row.top - pad * hei
            b = row.bottom + pad * hei
            return np.asarray([l, t, r-l, b-t])

        def restrict_box(box):
            l, t, w, h = box
            l = min(max(0, l), EPIC_SIZE[0])
            r = min(max(0, l + w), EPIC_SIZE[0])
            t = min(max(0, t), EPIC_SIZE[1])
            b = min(max(0, t + h), EPIC_SIZE[1])
            return np.asarray([l, t, r-l, b-t])

        def get_side(df, frame, side):
            entries = df[(df.frame == frame) & (df.det_type == 'hand') & (df.side == side)]
            if len(entries) == 0:
                return None
            det_hand_box = row2xywh(entries.iloc[0])
            det_hand_box = det_hand_box / (EPIC_HOA_SIZE * 2) * (EPIC_SIZE * 2)
            det_hand_box = restrict_box(det_hand_box)
            return det_hand_box
        left = get_side(self.df, frame, 'left')
        right = get_side(self.df, frame, 'right')
        if as_dict:
            return dict(left_hand=left, right_hand=right)
        else:
            return left, right
    
    def get_image(self, frame, as_pil=False) -> np.ndarray:
        if as_pil:
            return Image.open(f'{self.frames_root}/frame_{frame:010d}.jpg')
        return np.asarray(Image.open(f'{self.frames_root}/frame_{frame:010d}.jpg'))
    
    def visualise_hand_boxes(self, frame):
        odlib.setup('xywh')
        img = self.get_image(frame)
        hand_bbox_dict = self.get_hand_boxes(frame)
        img = odlib.draw_bboxes_image_array(img, hand_bbox_dict['left_hand'][None])
        return odlib.draw_bboxes_image_array(np.asarray(img), hand_bbox_dict['right_hand'][None])
    
    def frankmocap_forward(self, frame_start, frame_end, global_cam, side='left'):
        side_hand = 'left_hand' if side == 'left' else 'right_hand'

        hand_rot_6ds = []
        hand_translations = []
        mano_pca_poses = []
        
        for frame in tqdm.trange(frame_start, frame_end):
            img = self.get_image(frame)
            hand_bbox_dict = self.get_hand_boxes(frame, as_dict=True)
            if hand_bbox_dict[side_hand] is None:
                return None, None
            mocap_pred = self.hand_predictor.regress(img[..., ::-1], [hand_bbox_dict])
            one_hands = collate_mocap_hand(mocap_pred, side_hand)
            
            pred_hand_full_pose, pred_hand_betas, pred_camera = map(
                lambda x: torch.as_tensor(one_hands[x], device='cuda'),
                ('pred_hand_pose', 'pred_hand_betas', 'pred_camera'))
            hand_bbox_proc = one_hands['bbox_processed']
            rot_axisang = pred_hand_full_pose[:, :3]
            pred_hand_pose = pred_hand_full_pose[:, 3:]
            mano_pca_pose = recover_pca_pose(pred_hand_pose, side)

            hand_sz = torch.ones_like(global_cam.fx) * 224
            hand_cam = global_cam.crop(hand_bbox_proc).resize(new_w=hand_sz, new_h=hand_sz)
            hand_rotation_6d, hand_translation = compute_hand_transform(
                rot_axisang, pred_hand_pose, pred_camera, side, hand_cam=hand_cam)

            hand_rot_6ds.append(hand_rotation_6d)
            hand_translations.append(hand_translation)
            mano_pca_poses.append(mano_pca_pose)

        return hand_rot_6ds, hand_translations, mano_pca_poses
    
    def frankmocap_post(self, 
                        hand_rot_6ds: List, 
                        hand_translations: List, 
                        mano_pca_poses, 
                        c2ws: List,
                        frame_start,
                        frame_end,
                        scale_hand, 
                        side='left') -> List:

        # vh_cams = []
        fh = get_hand_faces(side)
        mh_worlds = []
        for frame in tqdm.trange(frame_start, frame_end):
            hand_rotation_6d = hand_rot_6ds[frame - frame_start].view(1, 6)
            hand_translation = hand_translations[frame - frame_start].view(1, 3)
            mano_pca_pose = mano_pca_poses[frame - frame_start].view(1, 45)
            c2w = c2ws[frame - frame_start]

            bsize = 1
            device = 'cuda'
            mano_rot = torch.zeros([bsize, 3], device=device)
            # mano_trans = torch.zeros([bsize, 3], device=device)
            mano_betas = torch.zeros([bsize, 10], device=device)
            mano_model = HomanManoModel('externals/mano', side=side)
            mano_model = mano_model.cuda()
            mano_res = mano_model.forward_pca(
                mano_pca_pose.view(bsize, 45), rot=mano_rot, betas=mano_betas)
            
            hand_rot_mat = rotation_6d_to_matrix(hand_rotation_6d)
            # front
            P28_101_magics = dict(
                left=dict(vscale=2.5, tscale=2.5, trans=torch.tensor([0, 0.23, 0], device='cuda', dtype=torch.float32)),
                right=dict(vscale=2.5, tscale=2.5, trans=torch.tensor([0, 0.20, 0], device='cuda', dtype=torch.float32))
                # right=dict(vscale=2.5, tscale=2.5, trans=torch.tensor([0, 0.20, 0.03], device='cuda', dtype=torch.float32))
            )
            magics = P28_101_magics

            verts_hand_og = mano_res['verts'] #+ mano_trans[None]  # hand space
            verts_hand_og = verts_hand_og * scale_hand * magics[side]['vscale']
            verts_hand_cam = verts_hand_og @  hand_rot_mat.permute(0, 2, 1) \
                + hand_translation * scale_hand * magics[side]['tscale'] \
                    + magics[side]['trans'] * magics[side]['tscale']

            mh_cam = SimpleMesh(verts_hand_cam, fh)
            mh_world = mh_cam.as_open3d.transform(c2w)
            mh_world.compute_vertex_normals()
            mh_world.compute_triangle_normals()
            mh_worlds.append(mh_world)

        return mh_worlds

    def frankmocap_pipeline(self, frame: int, global_cam, scale_hand=1.0, side='left',
                            viz_debug=False, verbose_debug=False, ret_trimesh=False):
        """ currently left hand
        Returns:
            vh: (778, 3) 
            fh: (F, 3) 
        """
        side_hand = 'left_hand' if side == 'left' else 'right_hand'
        img = self.get_image(frame)
        hand_bbox_dict = self.get_hand_boxes(frame, as_dict=True)
        if hand_bbox_dict[side_hand] is None:
            return None, None
        mocap_pred = self.hand_predictor.regress(img[..., ::-1], [hand_bbox_dict])
        one_hands = collate_mocap_hand(mocap_pred, side_hand)
        
        pred_hand_full_pose, pred_hand_betas, pred_camera = map(
            lambda x: torch.as_tensor(one_hands[x], device='cuda'),
            ('pred_hand_pose', 'pred_hand_betas', 'pred_camera'))
        hand_bbox_proc = one_hands['bbox_processed']
        rot_axisang = pred_hand_full_pose[:, :3]
        pred_hand_pose = pred_hand_full_pose[:, 3:]
        mano_pca_pose = recover_pca_pose(pred_hand_pose, side)

        hand_sz = torch.ones_like(global_cam.fx) * 224
        hand_cam = global_cam.crop(hand_bbox_proc).resize(new_w=hand_sz, new_h=hand_sz)
        hand_rotation_6d, hand_translation = compute_hand_transform(
            rot_axisang, pred_hand_pose, pred_camera, side, hand_cam=hand_cam)

        bsize = 1
        device = 'cuda'
        mano_rot = torch.zeros([bsize, 3], device=device)
        # mano_trans = torch.zeros([bsize, 3], device=device)
        mano_betas = torch.zeros_like(pred_hand_betas)
        mano_model = HomanManoModel('externals/mano', side=side)
        mano_model = mano_model.cuda()
        mano_res = mano_model.forward_pca(
            mano_pca_pose, rot=mano_rot, betas=mano_betas)
        verts_hand_og = mano_res['verts'] #+ mano_trans[None]  # hand space
        if verbose_debug:
            print(verts_hand_og.shape)

        hand_rot_mat = rotation_6d_to_matrix(hand_rotation_6d)
        P28_101_magics = dict(
            left=dict(vscale=2.5, tscale=2.5, trans=torch.tensor([0, 0.23, 0], device='cuda', dtype=torch.float32)),
            right=dict(vscale=2.5, tscale=2.5, trans=torch.tensor([0, 0.20, 0.03], device='cuda', dtype=torch.float32))
        )
        magics = P28_101_magics

        verts_hand_og = verts_hand_og * scale_hand * magics[side]['vscale']
        verts_hand_cam = verts_hand_og @  hand_rot_mat.permute(0, 2, 1) \
            + hand_translation * scale_hand * magics[side]['tscale'] \
                + magics[side]['trans'] * magics[side]['tscale']
        if verbose_debug:
            print('verts_hand_cam', verts_hand_cam.shape)
            print('hand_translation', hand_translation)
        fh = get_hand_faces(side)

        if ret_trimesh:
            return SimpleMesh(verts_hand_cam, fh)
        if viz_debug:
            # mesh = SimpleMesh(verts, faces)
            # return visualize_mesh(mesh, show_axis=True, viewpoint='nr')
            mesh = SimpleMesh(verts_hand_cam, fh)
            print(mesh.volume)
            rend = projection.perspective_projection_by_camera(
                mesh, global_cam, method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False), image=img)
            return rend
        return verts_hand_cam, fh
    
    def get_double_hand(self, frame, global_cam, cimg, scale_hand=1.0, ret_trimesh=False):
        c2w = colmap_image_c2w(cimg)

        left_o3d, right_o3d = None, None
        vl_cam, fh = self.frankmocap_pipeline(
            frame, global_cam=global_cam, scale_hand=scale_hand, side='left')
        if vl_cam is not None:
            left_cam = SimpleMesh(vl_cam, fh)
            if ret_trimesh:
                left_o3d = left_cam.apply_transform(c2w)
            else:
                # left_o3d = left_cam.apply_transform(c2w)
                # left_o3d = left_o3d.as_open3d
                left_o3d = left_cam.as_open3d.transform(c2w)
                left_o3d.compute_vertex_normals()
                left_o3d.compute_triangle_normals()

        vr_cam, fh = self.frankmocap_pipeline(
            frame, global_cam=global_cam, scale_hand=scale_hand, side='right')
        if vr_cam is not None:
            right_cam = SimpleMesh(vr_cam, fh)
            if ret_trimesh:
                right_o3d = right_cam.apply_transform(c2w)
            else:
                right_o3d = right_cam.as_open3d.transform(c2w)
                right_o3d.compute_vertex_normals()
                right_o3d.compute_triangle_normals()
        return left_o3d, right_o3d

