CUDA_VISIBLE_DEVICES=1
python predict_skin.py \
    --predict_data_list data/test_list.txt \
    --data_root data --model_name pct \
    --pretrained_model /home/hxgk/MoGen/jittor-comp-human/output/pct_ds2_0723/skin/best_model.pkl \
    --predict_output_dir predict/mix_online5_0.5 \
    --debug True \
    --sampler mix \
    --num_samples 2048 \
    --vertex_samples 1024 \
    --batch_size 64