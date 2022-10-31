# Colmap pipeline

# Config files

`app_default.ini` contains argument that are read directly from `--help`.

`custom.ini` contains customized arguments tuned based on `app_default`.

`generated_default.ini` contains arguments generated from `colmap generate_project`.

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

# Reproducing results from Vadim

- How to sample linearly? Ans: interval = int(max_frame_count / 1000), except P01_01.

# Colmap outputs

Colmap projects (intermediate outputs) will be stored in `./projects`.


# Interprete IMU data

- In https://github.com/epic-kitchens/VISOR-FrameExtraction:
https://raw.githubusercontent.com/epic-kitchens/VISOR-FrameExtraction/main/frames_to_timestamps.json

- unit: ACCL in m/s^2, GYRO in rad/s

# EPIC-KITCHENS-100 Settings

- GoPro model:      Hero 7
- Aspect Ratio:     16x9
- Resolution:       1080
- Frames Per Second:    50
- Field Of View:        Linear
- Video Stabilization:  ON
- GPS:                  OFF

From GoPro wesite:

- Focal Length for Linear FOV:  24mm (min) - 49mm (max)
- 16x9 Linear (Zoom = 0%) EIS ON:  
    - Vertical FOV = 56.7º  
    - Horizontal FOV = 87.6º  
    - Diagonal FOV = 95.5º

## Resources: 
- https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/AcquisitionGuidelines/GoProSettings_1.jpg
- https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/readme.txt
- https://gopro.com/help/articles/question_answer/hero7-field-of-view-fov-information?sf96748270=1