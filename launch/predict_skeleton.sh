CUDA_VISIBLE_DEVICES=7
python predict_skeleton.py \
       --predict_data_list data/test_list.txt \
       --data_root data --model_name pct \
       --pretrained_model output/mix_online5_0.5/skeleton/best_model.pkl \
       --predict_output_dir predict/mix_online5_0.5 \
       --debug True \
       --sampler mix \
       --num_samples 1024 \
       --vertex_samples 512 \
       --batch_size 256