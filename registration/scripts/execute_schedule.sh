#!/bin/bash
# This is based on `reg_script_loop_P04A01.sh`
# The argument is the index into 'schedule.txt'

LINE_IDX=$1
input_file=./registration/scripts/schedule.txt

line=$(sed "${LINE_IDX}q;d" "$input_file")

# # Split the line into an array
read -ra values <<< "$line"

# # Assign the values to variables
JOB="${values[0]}"
REF_VID="${values[1]}"
VIDS=("${values[@]:2}")
KID="${REF_VID:0:3}"

PROJECT_PATH="projects/registration/$JOB"
IMAGES_ROOT="/media/skynet/DATA/Datasets/epic-100/rgb/${KID}/"
SKELETONS=/home/skynet/Zhifan/epic_fields_full/skeletons

LOG_FILE=$PROJECT_PATH/log.txt

echo $PROJECT_PATH
echo $IMAGES_ROOT
echo "REF_VID = $REF_VID"

cp -r $SKELETONS/${REF_VID}_low/sparse/0 $PROJECT_PATH/model

for CUR_VID in ${VIDS[@]}
do
    echo "$CUR_VID Initialise"
    mkdir $PROJECT_PATH/$CUR_VID -p
    HOMO_TXT="/home/skynet/Ahmad/Zhifan_visualizer/build_epic_kitchens_3dmap/sampling/txt/${CUR_VID}/${CUR_VID}_list_homo_low.txt"
    IMAGE_LIST=$PROJECT_PATH/$CUR_VID/image_list.txt
    sed "s/^/${CUR_VID}\//" $HOMO_TXT > $IMAGE_LIST

    # Don't want to keep increasing database.db
    TMP_DB_PATH=/tmp/database.db
    cp $SKELETONS/${REF_VID}_low/database.db $TMP_DB_PATH

    echo "$CUR_VID Feature extract"
    colmap feature_extractor \
        --database_path $TMP_DB_PATH \
        --SiftExtraction.use_gpu 1 \
        --SiftExtraction.gpu_index 0 \
        --image_path $IMAGES_ROOT \
        --image_list_path $IMAGE_LIST \
        --ImageReader.existing_camera_id 1 > $LOG_FILE

    # Indexing will take a while, depending on the size of existing database
    # e.g. 4k images 4mins
    echo "$CUR_VID Matching"
    colmap vocab_tree_matcher \
        --database_path $TMP_DB_PATH \
        --SiftMatching.use_gpu 1 \
        --SiftMatching.gpu_index 0 \
        --VocabTreeMatching.vocab_tree_path /home/skynet/Zhifan/build_kitchens_3dmap/vocab_bins/vocab_tree_flickr100K_words256K.bin \
        --VocabTreeMatching.match_list_path $IMAGE_LIST > $LOG_FILE

    echo "$CUR_VID Image Registrate"
    colmap image_registrator \
        --database_path $TMP_DB_PATH \
        --input_path $PROJECT_PATH/model \
        --output_path $PROJECT_PATH/$CUR_VID > $LOG_FILE
done