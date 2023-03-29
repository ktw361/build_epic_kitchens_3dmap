# Registration
- Take a base video
- Choose 3 frames from a new video
- Register 3 frames into the base video
    - Get the R, T, s

Complexity linear to the number of videos

TODO: RANSAC, if the random 3 frames are accurate enough


# Registration (Unused Anymore)
Input: 
    - Given Single models (1, 2, 3, ..., n)
    - Given Mixed model containing images from (1, 2, 3, ... n)

1. Select the reference pose by smallest sum of reprojection error of subset images
    - Implement reprojection error calculation [DONE]
2. Get the reference pose (trivial)

# Storage
Registration is stored as a 3x3 rotation applying to col-vec, 1x3 translation and scalar scale.
```
model <- Read "model_path"  # (N, 3)
aligned_model = model * s * R.T + transl
```

# Verification Steps

1. Visualise Pcd (Coarse) [DONE]

2. Visualise Line Projection [Done]
    - Project line drawn on model_A to images of model_B

3. Hover camera of model_A over model_B