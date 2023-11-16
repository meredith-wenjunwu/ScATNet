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
# output_dir='../model/1scale/10x/49/super_melanocyte_area'
output_dir='../model/1scale/10x/49/no_attn_guide' # directory to save checkpoints
patience=5
scheduler='step'
use_gpu=1
gpu_id="0"
use_parallel="1"
train_split=6
aggregate_batch=1
lambda_attn=0.1
attn_loss="Frobenius" # "Frobenius" or "Inclusion_Exclusion"
optim='adam'


config_file='/projects/patho2/melanoma_diagnosis/model/temp/49_6144x12288_20231113-112941/config_resize_6144x12288_crop_512_train.json'

# python main.py \
# --seed $seed --self-attention --use-standard-emb --warmup \
# --data $data_dir --savedir $output_dir --base-extractor $base_extractor --weights $weights \
# --batch-size $batch_size --binarize --drop-out $dropout_rate --epochs $epochs \
# --attn $attn --attn_head $attn_head --lambda-attn $lambda_attn --attn-loss $attn_loss \
# --model-dir $output_dir --head-dim $head_dim --in-dim $in_dim --mode $mode --model $model --model-dim $model_dim \
# --num-classes 4 --n-layers $n_layers --lr $lr --lr-decay $lr_decay --loss-function $loss_fn \
# --num-crops $num_crops --patience $patience --resize1 $resize1 --scheduler $scheduler \
# --workers $workers --resize2 $resize2 --resize1-scale $resize1_scale --resize2-scale $resize2_scale \
# --use-parallel --aggregate-batch $aggregate_batch --use-gpu --gpu-id $gpu_id --weight-tie

python main.py --load-config $config_file --model-dir $output_dir --savedir $output_dir \
--attn $attn --attn_head $attn_head --lambda-attn $lambda_attn --attn-loss $attn_loss \
# --attn_guide 