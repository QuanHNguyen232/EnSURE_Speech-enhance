






# train cgan
MY_ID="test"
GENERATOR="unet"

CUDA_VISIBLE_DEVICES=0 python -u cgan_pix2pix.py \
                    --my_id $MY_ID --generator_type $GENERATOR \
                    > $f"./models_info/PD_unet_$MY_ID.txt"                    


# CUDA_VISIBLE_DEVICES=2 python -u cgan_cycle.py --my_id $MY_ID


