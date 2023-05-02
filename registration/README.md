# Registration
- Take a base video
- Choose 3 frames from a new video
- Register 3 frames into the base video
    - Get the R, T, s

Complexity linear to the number of videos


# Storage
Registration is stored as a 3x3 rotation applying to col-vec, 1x3 translation and scalar scale.
`aligned_model = model * s * R.T + transl`

e.g.
```projects/registration/P04A/P04A.json
[
    {
        "model_vid": "P04_01",
        "rot": [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ],
        "transl": [0.0, 0.0, 0.0],
        "scale": 1.0
    }
]
```

"line" field is stored in `projects/json_models/P04_01_skeletons.json`

Accessing {cameras.bin, images.bin, points.bin} can be done by `<model_prefix> + <model_vid> + <model_suffix>`;
Accessing image frames can be done by `<rgb_frames_root>/<model_vid[:3]>/<model_vid>/frames_%010d.jpg>`

# Usage
```
python registration/skeleton_registrator.py --infile json_files/registration/input/P01_in.json --step 1
python registration/skeleton_registrator.py --infile json_files/registration/input/P01_in.json --step 2
python registration/skeleton_registrator.py --infile json_files/registration/input/P01_in.json --step 3
```

# Verification Steps

0. Run skeleton_registration.py
    - This will register the model to the video
    - This will also save the registration to json
0.1: run rsync to sync json to local
1. Visualise Pcd (Coarse) [DONE]
2. Visualise Line Projection [DONE]
    - Project line drawn on model_A to images of model_B

# Verification Steps (Web)
1. `python tools/extract_primitives.py --model_dir ~/epic_fields_full/skeletons/P04_02_low/sparse/0 --out_file projects/json_models/P04_02_skeletons.json`