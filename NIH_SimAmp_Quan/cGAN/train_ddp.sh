






# train cgan
MY_ID="pix_ctx-loss_1"
GENERATOR="resnet"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 cgan_pix2pix.py \
                    --my_id $MY_ID --generator_type $GENERATOR --use_DDP \
                    > $f"./models_info/PD_$MY_ID.txt"

# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
#                     cgan_cycle.py --my_id $MY_ID --use_DDP \
#                     --n_steps 5 \
