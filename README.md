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
    └───sparse_images_medium  (854 x 480)
    └───sparse_images  (1920 x 1080)
    │   └───P01_103
    │   │  └───P01_103_frame_0000000140.jpg
    │   │  └───P01_103_frame_0000000298.jpg
    │   │  └───...
    │   └───P01_104
    │
    └───sparse_masks (854 x 480)
        └───P01_103
        │   └───P01_103_frame_0000000140.png
        │   └───P01_103_frame_0000000298.png
        │   └───...
        └───P01_103
```

- sparse_binary_masks: contains mask of size (1920, 1080)
- sparse_images_medium: resized version of images (854 x 480)

## Reproducing results from Vadim

- How to sample linearly? Ans: interval = int(max_frame_count / 1000), except P01_01.

## Colmap outputs

Colmap projects (intermediate outputs) will be stored in `./projects`.

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