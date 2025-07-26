CUDA_VISIBLE_DEVICES=7 python train_skeleton.py \
        --sampler_type mix --num_samples 1024 --vertex_samples 512 \
        --train_data_list data/train_list.txt \
        --val_data_list data/val_list.txt  --data_root data \
        --model_name pct \
        --bone_λ 0.1 --sym_λ 0.1  \
        --output_dir output/mix_bone_sym_0.1/skeleton \
        --epochs 300 --batch_size 256 \
        --scheduler step --lr_step 10 --learning_rate 1e-4 --lr_decay 0.95 \
        --optimizer adamw --weight_decay 0.0001

# CUDA_VISIBLE_DEVICES=2 python train_skin.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name pct --output_dir output/pct_ups2_0723/skin --batch_size 64 --lr_step 50 --lr_decay 0.5 \
#         --optimizer adamw