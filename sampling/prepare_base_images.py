"""
Input: kitchen id (e.g. P01)
Hyperparam: 
    - SAMPLE_FREQ: Sampling frequency. 
        - According to dissimilarity stats, this could be 20-50.
        - To trade-off speed, might have to set to 500.

Procedures:
    Find out what videos VISOR contains.

    For each video, e.g. P01_01, 
        - Each video has a FPS (either 50 or 60)
        i. find duration in seconds, hence num_frames = duration * FPS.
        ii. determine num_samples = num_frames / SAMPLE_FREQ.
        iii. apart from above fixed samples, add Visor frames.

Options
    1. Sample at a fixed frequency => bias towards longer videos
    2. Sample each video sample number of frames => belief that video contains equal amount of environment info 

TODO:
    [ ] use image_list instead of links
    [ ] image_list in sequential matching?
    [ ] allow partially mask
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import tqdm
import subprocess
from lib.utils import get_video_duration

visor_sparse_root = Path('/media/skynet/DATA/Zhifan/visor-sparse')
videos_dir = visor_sparse_root/'videos'
images_dir = visor_sparse_root/'images'

image_save_dir = Path('./projects/bases')


def prepare_images(pid: str, epic_set: str, SAMPLE_FREQ=100, resolution='854:480'):
    """
    Args:
        pid: e.g. P01
        epic_set: '100' or '55'
    """
    if epic_set == '100':
        FPS = 50.0
        num_digits = 3
    elif epic_set == '55':
        FPS = 59.94
        num_digits = 2
    else:
        raise ValueError("epic_set")
        
    all_vids = os.listdir(images_dir)
    related_vids = [v for v in all_vids 
                    if pid in v and len(v.split('_')[1]) == num_digits]
    related_vids = sorted(related_vids)
    total_frames, total_samples = 0, 0
    for vid in related_vids:
        video_file = videos_dir/(vid + '.MP4')
        dur = get_video_duration(video_file)
        num_frames = int(dur * FPS)
        num_samples = num_frames // SAMPLE_FREQ
        total_frames += num_frames
        total_samples += num_samples
        print(f'{vid} num_frames = {num_frames}\t(duration {int(dur)}s)\tnum_samples = {num_samples}')

        os.makedirs(image_save_dir/vid, exist_ok=True)
        pbar = tqdm.tqdm(total=num_samples)
        for sample_ind in range(1, num_samples+1):
            frame_ind = sample_ind * SAMPLE_FREQ
            time = frame_ind / FPS
            save_path = str(image_save_dir/vid/f'frame_{frame_ind:010d}.jpg')
            if os.path.exists(save_path):
                continue
            command = [
                'ffmpeg', 
                '-loglevel', 'error',
                '-threads', str(16),
                '-ss', str(time),
                '-i', str(video_file),
                '-qscale:v', '4', '-qscale', '2',
                '-vf', f'scale={resolution}',
                '-frames:v', '1',
                f'{save_path}'
            ]
            # print(' '.join(command))
            proc = subprocess.run(
                command,
                stdout=None,
                stderr=None,
                check=True, text=True)
            pbar.update(1)
        pbar.close()

        # TODO: add visor frames

    print(f'Total num_frames = {total_frames}, num_samples = {total_samples}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('pid')
    parser.add_argument('epic_set', choices=['100', '55'])
    parser.add_argument('--SAMPLE_FREQ', type=int, default=100)
    parser.add_argument('--resolution', type=str, default='854:480')
    args = parser.parse_args()
    prepare_images(args.pid, args.epic_set, args.SAMPLE_FREQ)