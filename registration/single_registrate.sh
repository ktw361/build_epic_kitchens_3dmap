PROJECT_PATH='projects/registration/P02A_bruteforce/'
TRY_VID='P02_02'
IMAGES=$PROJECT_PATH/$TRY_VID/images
IMAGE_LIST=$PROJECT_PATH/$TRY_VID/image_list.txt

find $IMAGES/ -name '*.jpg' | rev | cut -d/ -f1 | rev > $IMAGE_LIST

colmap feature_extractor \
    --database_path $PROJECT_PATH/database.db \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.gpu_index 0 \
    --image_path $IMAGES \
    --image_list_path $IMAGE_LIST \
    --ImageReader.existing_camera_id 1

# Indexing will take a while, depending on the size of existing database
# e.g. 4k images 4mins
colmap vocab_tree_matcher \
    --database_path $PROJECT_PATH/database.db \
    --SiftMatching.use_gpu 1 \
    --SiftMatching.gpu_index 0 \
    --VocabTreeMatching.vocab_tree_path /home/skynet/Zhifan/build_kitchens_3dmap/vocab_bins/vocab_tree_flickr100K_words256K.bin \
    --VocabTreeMatching.match_list_path $IMAGE_LIST
# colmap sequential_matcher \
#     --database_path $PROJECT_PATH/database.db \
#     --SiftMatching.use_gpu 1 \
#     --SiftMatching.gpu_index 0 \
#     --SequentialMatching.loop_detection 1 \
#     --SequentialMatching.vocab_tree_path /home/skynet/Zhifan/build_kitchens_3dmap/vocab_bins/vocab_tree_flickr100K_words256K.bin \

# use mapper instead of image_registrator + bundle_adjuster for more accurate results
# The Global Bundle Adjustment takes long (20mins-90mins)
# colmap mapper \
#     --database_path $PROJECT_PATH/database.db \
#     --image_path $IMAGES \
#     --input_path $PROJECT_PATH/model \
#     --output_path $PROJECT_PATH/$TRY_VID
colmap image_registrator \
    --database_path $PROJECT_PATH/database.db \
    --input_path $PROJECT_PATH/model \
    --output_path $PROJECT_PATH/$TRY_VID
    # --image_path $IMAGES \

echo $(date)