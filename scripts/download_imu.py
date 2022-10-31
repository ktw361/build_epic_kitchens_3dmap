import os
import os.path as osp
import subprocess


all_vids = ['P01_103', 'P01_104', 'P01_107', 'P02_101', 'P02_102', 'P02_107', 
            'P02_109', 'P02_121', 'P02_122', 'P02_124', 'P02_128', 'P02_130', 
            'P02_132', 'P02_135', 'P03_101', 'P03_112', 'P03_113', 'P03_120', 
            'P03_123', 'P04_101', 'P04_109', 'P04_110', 'P04_114', 'P04_121', 
            'P06_101', 'P06_102', 'P06_103', 'P06_106', 'P06_107', 'P06_108', 
            'P06_110', 'P07_101', 'P07_103', 'P07_110', 'P09_103', 'P09_104', 
            'P09_106', 'P11_101', 'P11_102', 'P11_103', 'P11_104', 'P11_105', 
            'P11_107', 'P12_101', 'P22_107', 'P22_117', 'P25_101', 'P25_107', 
            'P26_108', 'P26_110', 'P27_101', 'P27_105', 'P28_101', 'P28_103', 
            'P28_109', 'P28_110', 'P28_112', 'P28_113', 'P30_101', 'P30_107', 
            'P30_110', 'P30_111', 'P30_112', 'P35_105', 'P35_109', 'P37_101', 
            'P37_102', 'P37_103']

fmt = 'https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/%s/meta_data/%s-%s.csv'


def main(target_dir='./visor_data/imu'):
    os.makedirs(target_dir, exist_ok=True)
    sensors = ['accl', 'gyro']
    for vid in all_vids:
        pid = vid.split('_')[0]
        for sensor in sensors:
            url = fmt % (pid, vid, sensor)
            dst = f'{vid}-{sensor}.csv'
            dst = osp.join(target_dir, dst)
            if os.path.exists(dst):
                print(f"Skip downloaded {dst}.")
                continue
            subprocess.run([
                'wget', url, '-P', target_dir, '--show-progress',
            ], check=True, text=True)
    

if __name__ == '__main__':
    main()
