# Registration (Alignment of multiple models)


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

"line" is stored in `json_files/line_check`

Accessing {cameras.bin, images.bin, points.bin} can be done by `<model_prefix> + <model_vid> + <model_suffix>`;
Accessing image frames can be done by `<rgb_frames_root>/<model_vid[:3]>/<model_vid>/frames_%010d.jpg>`

# Usage

# Verification Steps

0. [TODO] script needed
1. Visualise line videos
2. Web visualise line and pcds