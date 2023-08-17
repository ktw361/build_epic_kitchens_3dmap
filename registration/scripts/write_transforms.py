from argparse import ArgumentParser
import tqdm
import os
import numpy as np
from libzhifan import io

from registration.functions import write_registration, umeyama_ransac
from registration.point_registrate import RegistrateManager, common_points3d


""" Run in batch
for i in (seq 1 45)
    python registration/scripts/write_transforms.py $i
end
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('idx', type=int, help='index into schedule.txt, start from 1')
    parser.add_argument('--schedule', default='registration/scripts/schedule.txt')
    return parser.parse_args()


# def get_cur_vid_list(ref_vid, kind, skeletons='/home/skynet/Zhifan/epic_fields_full/skeletons/'):
#     assert kind in {'A', 'B'}
#     kid = ref_vid[:3]
#     cur_vids = os.listdir(skeletons)
#     cur_vids = filter(lambda x: x.startswith(kid), cur_vids)
#     cur_vids = map(lambda x: x.replace('_low', ''), cur_vids)
#     vid_digits = 6 if kind == 'A' else 7
#     cur_vids = filter(lambda x: len(x) == vid_digits, cur_vids)
#     cur_vids = filter(lambda x: x != ref_vid, cur_vids)
#     return sorted(list(cur_vids))


def write_schedule(job_content: list, run_base_dir='./projects/registration'):
    """
    Args:
        job_content: a line in schedule.txt
            job, ref_vid, vid1, vid2, ... vidn
    """
    job, ref_vid, *cur_vids = job_content
    kind = job[-1]
    assert kind in {'A', 'B'}

    run_dir = os.path.join(run_base_dir, job)
    filename = os.path.join(run_dir, f'{job}.json')

    write_registration(filename, ref_vid, 1.0, np.eye(3), np.zeros(3))

    for cur_vid in tqdm.tqdm(cur_vids[:]):
        manager = RegistrateManager(ref_vid=ref_vid, cur_vid=cur_vid, run_dir=run_dir)
        retval = common_points3d(manager)
        if len(retval.common_images) == 0:
            print(f'{cur_vid} has no common images')
            continue
        xyz_ref = retval.xyz_ref
        xyz_cur = retval.xyz_cur
        s, R, t, _ = umeyama_ransac(xyz_cur, xyz_ref, k=500, t=0.1, d=0.25)
        if R is None:
            print(f'{cur_vid} has no good result')
            continue
        write_registration(filename, manager.cur_vid, s, R, t)


if __name__ == '__main__':
    args = parse_args()
    schedule = io.read_txt(args.schedule)
    write_schedule(schedule[args.idx-1])
    print(f"Job {args.idx} DONE")
    