model_dir='/projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/sma_guide_inclusion_exclusion/49_6144x12288_20231114-001931/'
config_name='config_cropsize_6144x12288_class_49_multi_resolution_train.json'
config_file="${model_dir}${config_name}"
resume_path="${model_dir}averaged_model_best7.pth"
python main.py --load-config $config_file --mode valid-train --resume $resume_path