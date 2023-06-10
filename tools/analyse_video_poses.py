from argparse import ArgumentParser
import os
import re
import numpy as np
import tqdm
import torch
from pyquaternion import Quaternion
import pickle
import pandas as pd
from lib.base_type import ColmapModel
from lib.common_functions import colmap_image_c2w, build_c2w_map_int
from registration.functions import (
    colmap_image_c2w
)
from libzhifan import io


def avg_quaternions_approx(quats: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        quats: (N, 4)
    Returns:
        qAvg: (4,)
    """
    if weights is not None and len(quats) != len(weights):
        raise ValueError("Args are of different length")
    if weights is None:
        weights = torch.ones_like(quats[:, 0])
    qAvg = torch.zeros_like(quats[0])
    for i, q in enumerate(quats):
        # Correct for double cover, by ensuring that dot product
        # of quats[i] and quats[0] is positive
        if i > 0 and torch.dot(quats[i], quats[0]) < 0.0:
            weights[i] = -weights[i]
        qAvg += weights[i] * q
    return qAvg / torch.norm(qAvg)


def compute_video_max_change(model: ColmapModel, from_mean=True, verbose=False,
                             max_only=False, num_samples: int=None):
    def quat_to_yaw_pitch_roll(q):
        yaw = np.arctan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z)
        pitch = np.arcsin(-2.0*(q.x*q.z - q.w*q.y))
        roll = np.arctan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
        return map(lambda x : x / np.pi * 180, (yaw, pitch, roll))
    
    time_fwd = build_c2w_map(model)
    frames = list(time_fwd.keys())

    if num_samples is None:
        ranger = range(len(frames))
    else:
        ranger = np.random.choice(len(frames), min(len(frames), num_samples), replace=False)

    if from_mean:
        origin_tvec = np.mean([v.tvec for v in time_fwd.values()], axis=0)
    else:
        origin_tvec = time_fwd[frames[0]].tvec
    max_d = np.zeros(3)
    infos = dict(xs=[], ys=[], zs=[], yaws=[], pitchs=[], rolls=[])
    for j in ranger:
        diff = time_fwd[frames[j]].tvec - origin_tvec
        max_d = np.maximum(diff, max_d)
        infos['xs'].append(diff[0])
        infos['ys'].append(diff[1])
        infos['zs'].append(diff[2])

    if from_mean:
        qvecs = torch.as_tensor([v.qvec for v in time_fwd.values()])
        origin_qvec = avg_quaternions_approx(qvecs).numpy()
    else:
        origin_qvec = time_fwd[frames[0]].qvec
    origin_qvec = Quaternion(origin_qvec)
    max_deg = np.zeros(3)
    for j in ranger:
        cur_qvec = time_fwd[frames[j]].qvec
        cur_qvec = Quaternion(cur_qvec)
        qvec_diff = origin_qvec * cur_qvec.inverse
        y, p, r = quat_to_yaw_pitch_roll(qvec_diff)
        max_deg = np.maximum(max_deg, [y, p, r])
        infos['yaws'].append(y)
        infos['pitchs'].append(p)
        infos['rolls'].append(r)

    if max_only:
        return max_d[0], max_d[1], max_d[2], max_deg[0], max_deg[1], max_deg[2]
    else:
        return infos
    
def to_cir_hist(angs, bins=36, log_scale=False):
    import matplotlib.pyplot as plt
    angs = np.asarray([v  if v > 0 else 360 + v for v in angs])
    cnt, theta = np.histogram(angs, range=(0, 360), bins=bins)
    if log_scale:
        cnt = np.log1p(cnt)
    
    N = len(cnt)
    theta = theta[:-1] / 360 * 2 * np.pi
    radii = cnt
    width = (2*np.pi) / N
    # N = 20    
    # theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    # radii = 10 * np.random.rand(N)

    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(theta, radii, width=width, bottom=0.0, color='blue')

    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax.set_xticklabels(map(lambda x : str(x) + '°', [0, 45, 90, 135, 180, -135, -90, -45]))
    # ax.set_xticklabels([str(label) +  for label in ax.get_xticks()])
    ax.set_yticklabels([])
    return ax


def rotation_main(save_name):
    # Sparse
    model_root = '/home/skynet/Zhifan/epic_fields_full/skeletons/'
    all_models = sorted(os.listdir(model_root))

    # df = dict(names=[], xs=[], ys=[], zs=[], yaws=[], pitches=[], rolls=[])
    df = dict(xs=[], ys=[], zs=[], yaws=[], pitches=[], rolls=[])
    for i, model_path in tqdm.tqdm(enumerate(all_models[:]), total=len(all_models)):
        full_model_path = os.path.join(model_root, model_path, 'sparse', '0')
        if not os.path.exists(full_model_path):
            continue
        model = ColmapModel(full_model_path)
        # x, y, z, yaw, pitch, roll = compute_video_max_change(model)
        infos = compute_video_max_change(
            model, from_mean=True, verbose=False, max_only=False, num_samples=5000)
        # print(f'{model_path}: {x:.3f}, {y:.3f}, {z:.3f}, {yaw:.3f}, {pitch:.3f}, {roll:.3f}')
        df['xs'].extend(infos['xs'])
        df['ys'].extend(infos['ys'])
        df['zs'].extend(infos['zs'])
        df['yaws'].extend(infos['yaws'])
        df['pitches'].extend(infos['pitchs'])
        df['rolls'].extend(infos['rolls'])
        # df['xs'].append(x)
        # df['ys'].append(y)
        # df['zs'].append(z)
        # df['yaws'].append(yaw)
        # df['pitches'].append(pitch)
        # df['rolls'].append(roll)

        if i % 100 == 0:
            pd.DataFrame(df).to_csv(f'{save_name}.csv', index=False)
    pd.DataFrame(df).to_csv(f'{save_name}.csv', index=False)
    
    
""" Translations """
def pca_calibration(obs, n_samples=10000):
    """ Zero-center the observations, project to the principle axes
    Args:
        obs: (N, 3)
    Returns:
        obs_calib: (N, 3)
    """
    obs = obs - obs.mean(0)  # estimate the mean using all
    N = len(obs)
    samples = obs[np.random.choice(N, min(N, n_samples), replace=False)]
    u, s, vh = torch.svd(torch.from_numpy(samples.T).cuda())
    u = u.cpu().numpy()
    obs = obs @ u.T
    return obs
def centering(obs):
    return obs - obs.mean(0)


def compute_video_translations(model: ColmapModel, from_mean=True, num_samples: int=None,
                               do_pca=False):
    time_fwd = build_c2w_map_int(model)
    frames = list(time_fwd.keys())
    tvecs = np.asarray([v.tvec for v in time_fwd.values()])

    if do_pca:
        tvecs = pca_calibration(tvecs, 10000)
    else:
        tvecs = centering(tvecs)

    if num_samples is None:
        ranger = range(len(frames))
    else:
        ranger = np.random.choice(len(frames), min(len(frames), num_samples), replace=False)

    if from_mean:
        origin_tvec = tvecs.mean(0)  # This should just be mean
        #np.mean([v.tvec for v in time_fwd.values()], axis=0)
    else:
        origin_tvec = time_fwd[frames[0]].tvec
    infos = dict(xs=[], ys=[], zs=[])
    for j in ranger:
        diff = tvecs[j] - origin_tvec
        infos['xs'].append(diff[0])
        infos['ys'].append(diff[1])
        infos['zs'].append(diff[2])

    return infos

def translation_main(save_name, do_pca, is_skeleton=True):
    # Sparse
    if is_skeleton:
        model_root = '/home/skynet/Zhifan/epic_fields_full/skeletons/'
    else:
        model_root = '/home/skynet/Zhifan/epic_fields_full/skeletons_extend/'
    all_models = sorted(os.listdir(model_root))

    df = dict(xs=[], ys=[], zs=[])
    for i, model_path in tqdm.tqdm(enumerate(all_models[:]), total=len(all_models)):
        if is_skeleton:
            full_model_path = os.path.join(model_root, model_path, 'sparse', '0')
        else:
            full_model_path = os.path.join(model_root, model_path)
        if not os.path.exists(full_model_path):
            continue
        model = ColmapModel(full_model_path)
        infos = compute_video_translations(
            model, from_mean=True, do_pca=do_pca)
        df['xs'].extend(infos['xs'])
        df['ys'].extend(infos['ys'])
        df['zs'].extend(infos['zs'])

        if i % 50 == 0:
            pd.DataFrame(df).to_csv(f'{save_name}.csv', index=False)
    pd.DataFrame(df).to_csv(f'{save_name}.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--save_name', type=str, default='video_translations_dense')
    parser.add_argument('--do_pca', action='store_true')
    args = parser.parse_args()
    save_name = args.save_name
    if args.do_pca:
        save_name = args.save_name + '_pca'
    translation_main(save_name, do_pca=args.do_pca, is_skeleton=False)
