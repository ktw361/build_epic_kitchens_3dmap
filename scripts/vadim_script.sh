source ~/.local/opt/anaconda3/etc/profile.d/conda.sh
conda activate colmap

export MULTIPLE_MODELS=1
export USE_GPU=0

DIR=$(realpath .)
ROOT=$(realpath $1)
DIR_VOCAB=$(realpath vocab)
PROJECT_PATH=$ROOT

cd $ROOT

echo $DIR_VOCAB
echo $PROJECT_PATH
echo $CURDIR

cd $PROJECT_PATH

echo $PROJECT_PATH $EXP_NAME $USE_GPU $OUT_DIR

colmap feature_extractor \
    --database_path database.db \
    --image_path frames \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu $USE_GPU \
    --ImageReader.mask_path masks \

# # # small: vocab_tree_flickr100K_words32K.bin, 100s to 1,000s of images
# # # medium:  vocab_tree_flickr100K_words256K.bin 1,000s to 10,000s of images

colmap vocab_tree_matcher \
    --database_path database.db \
    --VocabTreeMatching.vocab_tree_path $DIR_VOCAB/vocab_tree_flickr100K_words32K.bin \
    --SiftMatching.use_gpu $USE_GPU \
    # for some nodes num_threads too high, results in error:
    # `libgomp: Thread creation failed:`
    # reduce to 16
    # --SiftMatching.num_threads 16 \

mkdir sparse

colmap mapper \
    --database_path database.db \
    --image_path frames \
    --output_path sparse \
    --Mapper.num_threads 16 \
    --Mapper.init_min_tri_angle 4 \
    --Mapper.multiple_models $MULTIPLE_MODELS \
    --Mapper.extract_colors 1

cd $DIR