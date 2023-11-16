model_dir='/projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/no_attn_guide/'
config_name='config_cropsize_6144x12288_class_49_multi_resolution_train.json'
config_file="${model_dir}49_6144x12288_20231114-042419/${config_name}"
resume_path="${model_dir}best_model.pth"

python main.py --load-config $config_file --mode test --savedir $model_dir --save-result --resume $resume_path >> ${model_dir}test_result.txt