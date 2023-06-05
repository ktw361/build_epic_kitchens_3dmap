
VIDEO=$1 #i.e. P02_14
PRE=$2 #i.e. P02
GPU_IDX=$3 #0 or 1

skeleton_root='/home/skynet/Zhifan/epic_fields_full/skeletons/'
dense_cloud_root='./projects/colmap_models_cloud/'
# /media/skynet/DATA/Datasets/epic-100/rgb/P01/P01_04
#   --image_path /home/skynet/Zhifan/data/epic_rgb_frames/${PRE}/${VIDEO} \
colmap image_undistorter   \
   --image_path /media/skynet/DATA/Datasets/epic-100/rgb/${PRE}/${VIDEO} \
   --input_path ${skeleton_root}/${VIDEO}_low/sparse/0  \
   --output_path ${dense_cloud_root}/${VIDEO}   \
   --output_type COLMAP   \
   --max_image_size 2000 \


# slow
colmap patch_match_stereo    \
   --workspace_path ${dense_cloud_root}/${VIDEO}    \
   --workspace_format COLMAP    \
   --PatchMatchStereo.gpu_index=$GPU_IDX     \
   --PatchMatchStereo.cache_size=10 \

colmap stereo_fusion   \
   --workspace_path ${dense_cloud_root}/${VIDEO}   \
   --workspace_format COLMAP   \
   --input_type geometric   \
   --output_type BIN \
   --output_path ${dense_cloud_root}/${VIDEO} \
