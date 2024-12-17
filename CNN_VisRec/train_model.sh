SCRIPT_FLAGS="--method_type unet"
DATASET_FLAGS="--dataset galaxy --batch_size 128 --num_workers 6"
TRAIN_FLAGS="--microbatch 32 --save_interval 100 --max_step 2000 \
--model_save_dir ..."

python -m torch.distributed.launch --nproc_per_node=1 train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS