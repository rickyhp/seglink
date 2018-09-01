set -x
set -e
# ./scripts/train.sh 0 18 synthtext
export CUDA_VISIBLE_DEVICES=$1
IMG_PER_GPU=$2
DATASET=$3

export PYTHONPATH=/home/rp/code/pylib/src

CHKPT_PATH=${HOME}/code/seglink/models/seglink_icdar2015_512
TRAIN_DIR=${HOME}/code/seglink/models/seglink_icdar2015_512
#TRAIN_DIR=${HOME}/temp/no-use/seglink/seglink_icdar2015_384
#CHKPT_PATH=${HOME}/models/ssd-pretrain/seglink

# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
NUM_GPUS=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $NUM_GPUS \* $IMG_PER_GPU`

#dataset
if [ $DATASET == 'synthtext' ]
then
    DATA_PATH=SynthText
elif [ $DATASET == 'scut' ]
then
    DATA_PATH=SCUT
elif [ $DATASET == 'icdar2013' ]
then
    DATA_PATH=ICDAR
elif [ $DATASET == 'icdar2015' ]
then
    DATA_PATH=ICDAR
else
    echo invalid dataset: $DATASET
    exit
fi

DATASET_DIR=$HOME/code/icdar2015

python train_seglink.py \
			--train_dir=${TRAIN_DIR} \
			--num_gpus=${NUM_GPUS} \
			--learning_rate=0.0001 \
			--gpu_memory_fraction=-1 \
			--train_image_width=512 \
			--train_image_height=512 \
			--batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=${DATASET} \
            --dataset_split_name=train \
            --train_with_ignored=0 \
			--checkpoint_path=${CHKPT_PATH} \
			--using_moving_average=0
			
