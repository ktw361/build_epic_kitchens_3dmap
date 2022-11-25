# Colmap pipeline

# Config files

`custom.ini` contains customized arguments tuned based on `app_default`.

# Data structure

```
project
│   README.md
│
└───projects
│   └───<proj>
│
└───visor_data
    └───imu
    │   └───P01_103-accel.csv
    │   └───P01_103-gyro.csv
    │   └───...
    └───sparse_images_medium    (854 x 480)
    └───sparse_images           (VISOR origin, 1920 x 1080)
    │   └───P01_103
    │   │  └───P01_103_frame_0000000140.jpg
    │   │  └───P01_103_frame_0000000298.jpg
    │   │  └───...
    │   └───P01_104
    │
    └───sparse_binary_masks_medium  (854 x 480)
    └───sparse_binary_masks         (1920 x 1080)
    └───sparse_masks                (VISOR origin, Colored 854 x 480)
        └───P01_103
        │   └───P01_103_frame_0000000140.png
        │   └───P01_103_frame_0000000298.png
        │   └───...
        └───P01_103
```

## Image resolutions

Classify into 4 categories:
- vadim: 480 x 270
- small: 456 x 256
- medium: 854 x 480
- large: 1920 x 1080

Resolutions from different sources
- VISOR image: 1920 x 1080
- VISOR mask: 854 x 480
- Vadim copied: 480 x 270
- epic_rgb_frames: 456 x 256

# Type of masks
- No mask
- Simple mask, i.e. Vadim's mask
- Dynamic mask, i.e. VISOR mask but exclude unique objects
- VISOR mask (VISOR FULL mask)

unique_objects = 
    ['tap', 'fridge', 'hob', 'bin', 'oven', 'sink', 'dishwasher', 'freezer', 'machine washing', 'extractor fan']

# Colmap outputs

Colmap projects (intermediate outputs) will be stored in `./projects`.

# Interprete IMU data

- In https://github.com/epic-kitchens/VISOR-FrameExtraction:
https://raw.githubusercontent.com/epic-kitchens/VISOR-FrameExtraction/main/frames_to_timestamps.json

- unit: ACCL in m/s^2, GYRO in rad/s

# GoPro Settings

## EPIC-KITCHENS-100 Settings

- GoPro model:      Hero 7
- Aspect Ratio:     16x9
- Resolution:       1080
- Frames Per Second:    50
- Field of View:        Linear
- Video Stabilization:  ON
- GPS:                  OFF

From GoPro wesite:

- Focal Length for Linear FOV:  24mm (min) - 49mm (max)
- 16x9 Linear (Zoom = 0%) EIS ON:  
    - Vertical FOV = 56.7º  
    - Horizontal FOV = 87.6º  
    - Diagonal FOV = 95.5º

## EPIC-KITCHENS-55

- GoPro model:          Hero 5
- Frames Per Second:    59.94
- Field of View:        Linear
- Resolution:           1080p

Special cases:

* 1280x720: `P12_01`, `P12_02`, `P12_03`, `P12_04`.
* 2560x1440: `P12_05`, `P12_06` 
* 29.97 FPS: `P09_07`, `P09_08`, `P10_01`, `P10_04`, `P11_01`, `P18_02`,
    `P18_03`
* 48 FPS: `P17_01`, `P17_02`, `P17_03`, `P17_04`
* 90 FPS: `P18_09`

## Resources: 

- [EPIC-100 GoPro Setting](https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/AcquisitionGuidelines/GoProSettings_1.jpg)
- [EPIC-100 README](https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/readme.txt)
- https://gopro.com/help/articles/question_answer/hero7-field-of-view-fov-information?sf96748270=1
- [EPIC-55 README](https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/readme.txt)