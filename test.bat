set CUDA_VISIBLE_DEVICES=0
set CHECKPOINT_PATH="F:\CODE\dengdan\seglink\seglink-512\model.ckpt-217867"
rem set DATASET_DIR=F:\CODE\dengdan\seglink\ch4_test_images
set DATASET_DIR="F:\CODE\dengdan\seglink\alcohol_images"

python test_seglink.py --checkpoint_path=%CHECKPOINT_PATH% --gpu_memory_fraction=-1 --seg_conf_threshold=0.8 --link_conf_threshold=0.5 --dataset_dir=%DATASET_DIR%