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
└───colmap_projects
│   └───<proj>
│
└───visor_data
    └───imu
    │   └───P01_103-accel.csv
    │   └───P01_103-gyro.csv
    │   └───...
    └───sparse_images
    │   └───P01_103
    │   │  └───P01_103_frame_0000000140.jpg
    │   │  └───P01_103_frame_0000000298.jpg
    │   │  └───...
    │   └───P01_104
    │
    └───sparse_masks
        └───P01_103
        │   └───P01_103_frame_0000000140.png
        │   └───P01_103_frame_0000000298.png
        │   └───...
        └───P01_103
```

## Colmap outputs

Colmap projects (intermediate outputs) will be stored in `./colmap_projects`.

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