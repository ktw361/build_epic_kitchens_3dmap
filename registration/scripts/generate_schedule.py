import pandas as pd
# Generate shcedule.txt
change_infos = pd.read_csv('./extra/Extension_Participants.csv')
skl_infos = pd.read_csv('./outputs/skeleton_model_infos.csv')
pids = sorted(list(set(map(lambda x: x.split('_')[0], skl_infos.vid)))) # get pids
all_vids = list(skl_infos.vid)

groups = []
for pid in pids:
    if pid not in set(change_infos['Participant']):
        # only in 55 => A
        vids = [v for v in all_vids if v.startswith(pid)]
        groups.append([pid, 'A'] + vids)
        continue
    entry = change_infos[change_infos.Participant == pid]
    assert len(entry) == 1
    entry = entry.iloc[0]
    if entry.NewParticipant == 1:
        # only in 100 => B
        vids = [v for v in all_vids if v.startswith(pid)]
        groups.append([pid, 'B'] + vids)
    elif entry.ChangingKitchen == 1:
        # need A and B
        vids = [v for v in all_vids if v.startswith(pid) and len(v) == 6]
        groups.append([pid, 'A'] + vids)
        vids = [v for v in all_vids if v.startswith(pid) and len(v) == 7]
        groups.append([pid, 'B'] + vids)
    elif entry.ReturningKitchen == 1:
        # AB
        vids = [v for v in all_vids if v.startswith(pid)]
        groups.append([pid, 'AB'] + vids)
    else:
        raise ValueError('Not allowed')
    
points_info = {row.vid: row.points for _, row in skl_infos.iterrows()}
schedule = []  # each line is ['PxxA', ref_vid, vid_1, vid_2, ...]
for group in groups:
    pid, suf, *vids = group
    ref_vid = None
    max_num_points = 0
    for vid in vids:
        if points_info[vid] > max_num_points:
            ref_vid = vid
            max_num_points = points_info[vid]
    schedule.append([pid + suf, ref_vid] + [v for v in vids if v != ref_vid])

io.write_txt([' '.join(v) for v in schedule], 'scratch/shedule.txt')