# Registration (Proposal)
Input: 
    - Given Single models (1, 2, 3, ..., n)
    - Given Mixed model containing images from (1, 2, 3, ... n)

1. Select the reference pose by smallest sum of reprojection error of subset images
    - Implement reprojection error calculation [DONE]
2. Get the reference pose (trivial)

# Data [Required]
What's the efficient way to get Mixed model? Can simply take the union? e.g. P04 has 56 videos (returning kitchen)
- will hierarchical processing affect accuracy?

# Storage
Registration is stored as a 3x3 rotation applying to col-vec, 1x3 translation and scalar scale.
```
model <- Read "model_path"  # (N, 3)
aligned_model = model * s * R.T + transl
```

# Verification Steps

1. Visualise Pcd (Coarse) [DONE]

2. Visualise Line Projection
    - Project line drawn on model_A to images of model_B

3. Hover camera of model_A over model_B