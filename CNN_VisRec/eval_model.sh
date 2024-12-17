SCRIPT_FLAGS="--method_type unet"
DATASET_FLAGS="--dataset galaxy \
--batch_size 1 --num_workers 2"
TEST_FLAGS="--model_save_dir ... --resume_checkpoint ... \
--output_dir ... \
--debug_mode False"

python -m torch.distributed.launch --nproc_per_node=1 test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS

python myfft.py
