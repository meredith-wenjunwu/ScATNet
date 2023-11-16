#Base feature extractor
base_extractor="mobilenetv2"
weights="model/pretrained_cnn_models/mobilenetv2_s_1.0_imagenet_224x224.pth" # feature extractor weights

# Location of the data
data_dir="/projects/patho2/melanoma_diagnosis/x10/binarized/49/"
# data_dir="/projects/patho2/melanoma_diagnosis/x10/multi_scale/3scale/"
attn="/projects/patho4/Kechun/diagnosis/melanocyte_attention/dataset/attention_maps/super_melanocyte_area"

# data loader setting
workers=2
binarize=0
num_crops=7
resize1="6144 12288"
resize2="512 512"
# resize1_scale="0.0 1.0 2.0"
resize1_scale="2.0"
resize2_scale="1.0"


# Model details
model='multi_resolution'
dropout_rate=0.25
head_dim=32
in_dim=1280
model_dim=128
n_layers=4
attn_head=2

# experiment setup
seed=1669
mode='train'
lr=0.0005
lr_decay=0.5
loss_fn='bce'
batch_size=12
eval_split=1
epochs=200 # Number of training epochs
# output_dir='../model/1scale/10x/49/no_attn_guide'
output_dir='../model/1scale/10x/49/super_melanocyte_area/inclusion_exclusion' # directory to save checkpoints
patience=5
scheduler='step'
use_gpu=1
gpu_id="1"
use_parallel="1"
train_split=6
aggregate_batch=8
lambda_attn=0.1
attn_loss="Inclusion_Exclusion" # "Frobenius" or "Inclusion_Exclusion"
optim='adam'

# python main.py \
# --seed $seed \
# --data $data_dir --savedir $output_dir --base-extractor $base_extractor --weights $weights \
# --batch-size $batch_size --binarize --drop-out $dropout_rate --epochs $epochs \
# --attn $attn --attn_head $attn_head --attn_guide --lambda-attn $lambda_attn --attn-loss $attn_loss \
# --model-dir $output_dir --head-dim $head_dim --in-dim $in_dim --mode $mode --model $model --model-dim $model_dim \
# --num-classes 4 --n-layers $n_layers --lr $lr --lr-decay $lr_decay --loss-function $loss_fn \
# --num-crops $num_crops --patience $patience --resize1 $resize1 --scheduler $scheduler \
# --workers $workers --resize2 $resize2 --resize1-scale $resize1_scale --resize2-scale $resize2_scale \
# --use-parallel --aggregate-batch $aggregate_batch --use-gpu --gpu-id $gpu_id --weight-tie

python main.py --load-config /projects/patho2/melanoma_diagnosis/model/temp/49_6144x12288_20231113-112941/config_resize_6144x12288_crop_512_train.json --model-dir $output_dir --savedir $output_dir --attn $attn --attn_head $attn_head --lambda-attn $lambda_attn --attn-loss $attn_loss --attn_guide 


# ## average good models
# python utilities/average_checkpoints.py --checkpoint-dir /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148 --best-n 3

# python utilities/average_checkpoints.py --checkpoint-dir /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148 --best-n 5

# python utilities/average_checkpoints.py --checkpoint-dir /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148 --best-n 7

# python model_select.py --checkpoint-dir /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148 --load-config /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148/config_cropsize_6144x12288_class_49_multi_resolution_train.json

# ## validation on train set
# python main.py --load-config /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148/config_cropsize_6144x12288_class_49_multi_resolution_train.json --mode valid-train --resume /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148/averaged_model_best3.pth

# ## test
# python main.py --load-config /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148/config_cropsize_6144x12288_class_49_multi_resolution_train.json --mode test --resume /projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/super_melanocyte_area/49_6144x12288_20231113-151148/averaged_model_best7.pth

# # -------------Test Report----------------
# #               precision    recall  f1-score   support

# #            0     0.5000    0.9118    0.6458        68
# #            1     0.4694    0.3485    0.4000        66
# #            2     0.4861    0.4667    0.4762        75
# #            3     0.7407    0.3175    0.4444        63

# #     accuracy                         0.5147       272
# #    macro avg     0.5491    0.5111    0.4916       272
# # weighted avg     0.5445    0.5147    0.4928       272

# # TN, FP, FN, TP 62 2 35 23
# # Test acc: 0.5147
# # [[62  2  4  0]
# #  [35 23  7  1]
# #  [19 15 35  6]
# #  [ 8  9 26 20]]