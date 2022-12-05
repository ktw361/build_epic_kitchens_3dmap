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

"""

from argparse import ArgumentParser
import os
from pathlib import Path
import tqdm
import subprocess
from lib.utils import get_video_duration
from libzhifan import io

videos_dir = Path('/media/skynet/DATA/Zhifan/visor-sparse/videos')
visor_images_dir = Path('./visor_data/sparse_images_medium/')

image_save_dir = Path('./visor_data/sampled_frames')
list_save_dir = Path('./sampling/txt')


def prepare_images(pid: str, epic_set: str, SAMPLE_FREQ=100, resolution='854:480',
                   print_info=False):
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

    all_vids = os.listdir(visor_images_dir)
    related_vids = [v for v in all_vids
                    if pid in v and len(v.split('_')[1]) == num_digits]
    related_vids = sorted(related_vids)
    total_frames, total_samples = 0, 0

    image_list = []
    mask_mapping = []

    for vid in related_vids:
        video_file = videos_dir/(vid + '.MP4')
        dur = get_video_duration(video_file)
        num_frames = int(dur * FPS)
        num_samples = num_frames // SAMPLE_FREQ
        total_frames += num_frames
        total_samples += num_samples
        print(f'{vid} num_frames = {num_frames}\t(duration {int(dur)}s)\tnum_samples = {num_samples}')

        os.makedirs(image_save_dir/vid, exist_ok=True)
        pbar = tqdm.tqdm(total=num_samples, disable=print_info)
        for sample_ind in range(1, num_samples+1):
            frame_ind = sample_ind * SAMPLE_FREQ
            time = frame_ind / FPS
            relative_frame_name = Path(vid)/f'frame_{frame_ind:010d}.jpg'
            save_path = str(image_save_dir/relative_frame_name)
            image_list.append(os.path.join('sampled_frames', relative_frame_name))

            if os.path.exists(save_path):
                continue
            if print_info:
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
            subprocess.run(
                command,
                stdout=None,
                stderr=None,
                check=True, text=True)
            pbar.update(1)
        pbar.close()

        # Add visor frames
        visor_frames = sorted(os.listdir(visor_images_dir/vid))
        for visor_frame in visor_frames:
            frame_name = os.path.join('sparse_images_medium', vid, visor_frame)
            mask_name = os.path.abspath(os.path.join(
                'visor_data/sparse_binary_masks_medium', vid,
                visor_frame.replace('jpg', 'png')))
            image_list.append(frame_name)
            mask_mapping.append([frame_name, mask_name])

    print(f'Total num_frames = {total_frames}, num_samples = {total_samples}')
    if not print_info:
        os.makedirs(list_save_dir/pid, exist_ok=True)
        io.write_txt(image_list, list_save_dir/pid/f'image_list_freq{SAMPLE_FREQ}.txt')
        mask_mapping_text = [' '.join(v) for v in mask_mapping]
        io.write_txt(mask_mapping_text, list_save_dir/pid/f'mapping_freq{SAMPLE_FREQ}.txt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('pid')
    parser.add_argument('epic_set', choices=['100', '55'])
    parser.add_argument('--SAMPLE_FREQ', type=int, default=100)
    parser.add_argument('--resolution', type=str, default='854:480')
    parser.add_argument('--print-info', default=False, action='store_true')
    args = parser.parse_args()
    prepare_images(args.pid, args.epic_set, args.SAMPLE_FREQ, print_info=args.print_info)
