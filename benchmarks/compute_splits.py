import tqdm
import os
import os.path as osp
import re
import numpy as np
import pandas as pd
from lib.base_type import ColmapModel
import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='outputs/probe_secs_90.log',
                    filemode='a',
                    level=logging.DEBUG)
logger = logging.getLogger()

SEED = 0

# TODO: VISOR to EPIC frame mapping


def get_FPS(vid: str) -> float:
    if vid in {'P09_07', 'P09_08', 'P10_01', 'P10_04', 'P11_01', 'P18_02', 'P18_03'}:
        return 29.97
    elif vid in {'P17_01', 'P17_02', 'P17_03', 'P17_04'}:
        return 48.0
    elif vid == 'P18_09':
        return 90.0
    elif len(vid) == 6: # EPIC-55
        """ Although it's 59.94, the table seems to do the computation with 60FPS
        """
        return 60.0
    elif len(vid) == 7: # EPIC-100
        return 50.0
    else:
        raise ValueError(f"vid {vid} bad.")


class StampManager:
    """ currently no control of final act/noact ratio"""

    def __init__(self,
                 stats,
                 pool_size,
                 min_train_size,
                 debug=False):
        self.tot_iamges = len(stats)

        noact_size, action_size = pool_size // 2, pool_size // 2
        # test = action + noact
        _action_rows = stats[stats.nid != ''].sample(action_size, random_state=SEED)
        _noact_rows = stats[stats.nid == ''].sample(noact_size, random_state=SEED)
        self.test_rows = pd.concat(
            [_action_rows, _noact_rows], ignore_index=True, sort=False)
        self.test_rows.sort_values(by='t', inplace=True)
        _train_rows = stats[~stats.index.isin(self.test_rows.index)]
        self.train_rows = [v for i, v in _train_rows.iterrows()]
        self.train_rows.sort(key=lambda x: x.t)

        # windows: (start, end, rows-in-this-window)
        self.init_stamps = [
            (v.t, v.t, [v]) for i, v in self.test_rows.iterrows()]

        self.min_train_size = min_train_size
        self.debug = debug

    @staticmethod
    def get_total_window_rows(windows):
        res = []
        for win in windows:
            res += win[2]
        return res

    def _merge(self, r: float):
        """ extend (s, e) to (s-ext, t+ext)
        Then merge to earliest window.
        """
        # Step1: generate window
        windows = [None for _ in range(len(self.init_stamps))]
        for i, (t1, t2, rows) in enumerate(self.init_stamps):
            windows[i] = (t1 - r, t2 + r, rows)
        # Step2: merge windows
        new_windows = []
        for i, (t1, t2, rows) in enumerate(windows):
            if i == 0:
                new_windows.append((t1, t2, rows))
            else:
                s, e, old_rows = new_windows[-1]
                if t1 <= e:
                    new_windows[-1] = (s, t2, old_rows + rows)
                else:
                    new_windows.append((t1, t2, rows))
        windows = new_windows
        if self.debug:
            print(f'num_test_rows after merge = {len(self.get_total_window_rows(windows))}')
        return windows

    def _rebalance(self, windows):
        j = 0
        keep_inds = []
        discards = [[] for _ in range(len(windows))]
        for i, train_row in enumerate(self.train_rows):
            t = train_row.t
            while j < len(windows) and windows[j][1] < t:
                j += 1
            if j == len(windows):
                keep_inds.append(i)
                break
            if windows[j][0] <= t:
                discards[j].append(train_row)
            else:
                keep_inds.append(i)

        train_rows = [self.train_rows[i] for i in keep_inds]

        if self.debug:
            num_discards = sum([len(v) for v in discards])
            print(f'before rebalacing: {len(train_rows)} train rows, {num_discards} discards')

        # Now, we have to give back to training set
        np.random.seed(SEED)
        while len(train_rows) < self.min_train_size:
            if len(windows) == 0:
                break
            win_id = np.random.choice(len(windows))
            revive = windows.pop(win_id)
            train_rows += revive[2]
            revive2 = discards.pop(win_id)
            train_rows += revive2
        train_rows.sort(key=lambda x: x.t)

        return train_rows, windows

    def probe(self, r: float):
        """ a stateless probe """

        windows = self._merge(r)
        train_rows, windows = self._rebalance(windows)
        test_rows = self.get_total_window_rows(windows)

        if self.debug:
            # check if there are really so many leaveout rows inside this window
            leaveouts = []
            for row in self.train_rows:
                for win in windows:
                    if win[0] <= row.t <= win[1]:
                        leaveouts.append(row)
            for i, row in self.test_rows.iterrows():
                for win in windows:
                    if win[0] <= row.t <= win[1]:
                        leaveouts.append(row)
            print(f'Leaveout from train size = {len(leaveouts)}')

        return train_rows, windows, test_rows


class SplitGenerator:

    def __init__(self,
                 model_dir: str,
                 test_ratio=0.1,
                 inlier_ratio=0.5,
                 ):
        """
        Say we have N registered timestamps, we want r_test = 10% test timestamps,
        but we know inlier ratio is at least r_inlier = 50% (in practice it's higher), and we haven't draw lines so
        we don't yet know which frames are incorrect; (we can't generate lines arbitrarily because that line might be generate through errorneous poses)
        So, we first sample r_large_test timestamps, s.t r_test = r_large_test * r_inlier, i.e 10% = 20% * 50%,
        then, after we check the line-error, we sample r_inlier timestamps in the r_large_test,
        finally we 'give-back' to training set #-r_large_test - #-r_test timestamps.
        If we believe actually we don't have any errors in pose, we can immediately give-back.

        Args:
            model_dir: <path to dir> that contains images.bin. Must contain vid identifiers
        """
        self.model = ColmapModel(model_dir)
        self.vid = re.search('P\d{2}_\d{2,3}', model_dir)[0]
        self.FPS = get_FPS(self.vid)
        self.test_ratio = 0.1
        self.inlier_ratio = 0.5
        self.pool_ratio = test_ratio / inlier_ratio  # r_large_test

    def get_frame_index(self, frame_name: str) -> int:
        frame = re.search('\d{10,}', frame_name)[0]
        return int(frame)

    def convert_sec(self, frame_name: str):
        frame = float(re.search('\d{10,}', frame_name)[0])
        return frame / self.FPS

    def compute_stats(self, train_df):
        """
        Returns:
            <img_name, t, action_row, err>
        """
        img_names = []
        ts = []
        narration_id = []
        errs = []
        df = train_df[train_df['video_id'] == self.vid]
        for img in self.model.ordered_images:
            name = img.name
            frame_ind = self.get_frame_index(name)
            t = frame_ind / self.FPS
            """ it's numerically safer to locate action by frame """
            entries = df[(df.start_frame <= frame_ind) & (frame_ind <= df.stop_frame)]
            if len(entries) == 0:
                nid = ''
            else:
                nid = entries.iloc[0].narration_id
            err = -1
            img_names.append(name)
            ts.append(t)
            narration_id.append(nid)
            errs.append(err)

        stats = {
            'img_name': img_names,
            't': ts,
            'nid': narration_id,
            'err': errs
        }
        return pd.DataFrame(stats)

    def get_curve(self, stats):
        """
        Starting from half-window size r = 0.0, we can select exactly num_frames * r_large_test frames;
        when we gradually increase half-window size, we have many test frames falling into the same window,
        plus training images falling into the same window.
        When minimum training number is violated, we drop random test timestamp until condition satisfied again.

        We can further apply NO_SAME_ACTION constraint, this will even decrease available test frames though.

        Returns:
            -probes
            -windows:
            -test_rows

        """
        tot_images = self.model.num_images
        pool_size = int(tot_images * self.pool_ratio)
        min_train_size = tot_images - pool_size
        """ noact_size, action_size = pool_size // 2, pool_size // 2 """
        if len(stats[stats.nid != '']) == 0:
            return None, None, None
        stamps = StampManager(stats, pool_size, min_train_size)

        rs = np.linspace(0, 3, num=20)
        nr, nt, nw = [], [], []
        for r in rs:
            train_rows, windows, test_rows = stamps.probe(r)
            nr.append(len(train_rows))
            nt.append(len(test_rows))
            nw.append(len(windows))

        return rs, nw, nt

    def save_curve(self, stats, out_dir):
        plt.title(f'{self.vid} tot = {self.model.num_images}')
        rs, nw, nt = self.get_curve(stats)
        if rs is None:
            return
        # epylab.plot(rs, nr, '-x')
        plt.plot(rs, nt, '-x')
        plt.plot(rs, nw, '-x')
        plt.show()
        out_path = osp.join(out_dir, f'{self.vid}.png')
        plt.savefig(out_path)
        plt.clf()

    def giveback(self, errs):
        pass

    def visualize(self):
        """ What to visualize ? """
        pass


if __name__ == '__main__':
    df = pd.read_csv('/home/skynet/Zhifan/data/epic/EPIC_100_train.csv')

    # gen = SplitGenerator('projects/videos2023/P01_01/model/')
    out_dir = 'outputs/probe_secs_90'
    root = 'projects/videos2023/'
    videos = os.listdir(root)
    os.makedirs(out_dir, exist_ok=True)
    for video in tqdm.tqdm(videos):
        gtor = SplitGenerator(
            model_dir=osp.join(root, video, 'model'),
            test_ratio=0.1,
            inlier_ratio=1.0)  # 0.1
        stats = gtor.compute_stats(df)
        if len(stats[stats.nid != '']) == 0:  # no-action
            logger.info(f'No action in {video}')
            continue
        gtor.save_curve(stats, out_dir=out_dir)
