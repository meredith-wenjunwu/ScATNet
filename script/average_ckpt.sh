checkpoint_dir='/projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/sma_guide_inclusion_exclusion/49_6144x12288_20231114-001931'

python utilities/average_checkpoints.py --checkpoint-dir $checkpoint_dir --best-n 3

python utilities/average_checkpoints.py --checkpoint-dir $checkpoint_dir --best-n 5

python utilities/average_checkpoints.py --checkpoint-dir $checkpoint_dir --best-n 7