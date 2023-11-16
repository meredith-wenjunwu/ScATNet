for directory in `find /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/sma_guide_frobenius -type d -maxdepth 1 -mindepth 1`
do
    echo $directory
    checkpoint_dir=$directory
    python utilities/average_checkpoints.py --checkpoint-dir $checkpoint_dir --best-n 3
    python utilities/average_checkpoints.py --checkpoint-dir $checkpoint_dir --best-n 5
    python utilities/average_checkpoints.py --checkpoint-dir $checkpoint_dir --best-n 7
done