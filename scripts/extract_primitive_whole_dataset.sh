#!/bin/bash

# source folder format Pxx_yyy_low/{cameras,images,points3D}.bin
# destination format Pxx_yyy_skeletons_extend.json

registered_root=/media/deepthought/DATA/Ahmad/colmap_models_registered/
target_root=/media/skynet/DATA/Zhifan/colmap_projects/json_models/

for vid in $(ls $registered_root | sort); do 
    # Starts with P, followed by 2 digits, followed by _, followed by 2 or 3 digits followed by anything
    if [[ ! $vid =~ ^P[0-9]{2}_[0-9]{2,3}.*$ ]]; then
        continue
    fi
    # Check if the folder contains the 3 files
    folder=$registered_root/$vid
    if [[ ! -f "$folder/cameras.bin" || ! -f "$folder/images.bin" || ! -f "$folder/points3D.bin" ]]; then
        continue
    fi
    dest=$(echo "$vid" | sed 's/\(P[0-9]\{2\}_[0-9]\{2,3\}\)_low/\1_skeletons_extend/')
    dest=$target_root/$dest.json
    if [[ -f "$dest" ]]; then
        continue
    fi
    # Check target does not exist
    echo "$vid => $dest"
    python tools/extract_primitives.py --model_dir $folder  --out_file $dest
done