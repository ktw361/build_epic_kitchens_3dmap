# merge images and masks, copy database, then continue here with new registration
# images: contains ALL images (initial and extended)
# database: database from initial reconstruction
# PROJECT_PATH=projects/debug_insert2/
PROJECT_PATH=projects/P01-visor/
NEW_FRAMES=$PROJECT_PATH/new_frames/
NEW_OUTPUT=$PROJECT_PATH/sparse/new_res
USE_GPU=1

mkdir $NEW_OUTPUT -p

colmap feature_extractor \
    --database_path $PROJECT_PATH/database.db \
    --image_path $NEW_FRAMES \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu $USE_GPU \
    # --ImageReader.mask_path $PROJECT_PATH/masks \

# use larger dictionary
colmap vocab_tree_matcher \
    --database_path $PROJECT_PATH/database.db \
    --VocabTreeMatching.vocab_tree_path vocab_bins/vocab_tree_flickr100K_words256K.bin \
    --SiftMatching.use_gpu $USE_GPU \
# colmap sequential_matcher \
#     --SiftMatching.use_gpu $USE_GPU \
#     --database_path $PROJECT_PATH/database.db

colmap image_registrator \
    --database_path $PROJECT_PATH/database.db \
    --input_path $PROJECT_PATH/sparse/0 \
    --output_path $NEW_OUTPUT

# colmap bundle_adjuster \
#     --input_path $PROJECT_PATH/new_res \
#     --output_path $PROJECT_PATH/new_res_ba
