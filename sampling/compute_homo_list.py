import os
import subprocess
from libskynet import epic_rgb, visor_sparse

my_pid = ['P01', 'P09', 'P13', 'P31', 'P32', 'P33', 'P34', 'P35', 'P36', 'P37']
need_vids = [v for v in epic_rgb.avail_vids if v[:3] in my_pid]

for vid in need_vids:
    P = vid[:3]
    dst_file = f'/home/skynet/Zhifan/build_kitchens_3dmap/sampling/txt/{vid}/image_list_homo90.txt'
    if os.path.exists(dst_file):
        continue
    command = [
        'python', 'homography_filter/filter.py',
        '--src', f'/home/skynet/Zhifan/data/epic_rgb_frames/{P}/{vid}',
        '--dst_file', dst_file,
        '--overlap', '0.90']
    command = ' '.join(command)
    print(command)
    os.system(command)
    # subprocess.run(command)