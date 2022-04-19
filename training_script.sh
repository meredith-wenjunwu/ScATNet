
#Base feature extractor
base_extractor="mobilenetv2"
weights="model/pretrained_cnn_models/mobilenetv2_s_1.0_imagenet_224x224.pth" # feature extractor weights


# dataset
data="sample_data"
n_classes=4
num_crops="7 11" # number of crops per scale
resize1="6144 12288" # size to resize input images to
resize1_scale="0.375 0.625" # chosen scales relative to original image (0.375 * 20=7.5x)
transform="Zooming"
workers=4
channels=3


# model setting
drop_out=0.2
model="multi_resolution"
model_dim=128
model_dir="./model_zoo/"
head_dim=32
in_dim=1280
n_layers=4
linear_channel=4
num_scale_attn_layer=2

# general setting
aggregate_batch=8
batch_size=1
gpu_id="0 1"
loss_function='bce'
lr=0.0005
lr_decay=0.5
patience=50
max_bsz_cnn_gpu0=5 # maximum number of crops on each gpu
mode="train"
optim="adam"
savedir="./"
scheduler="step"
weight_decay=4e-06
epochs=200
start_epoch=0




CUDA_VISIBLE_DEVICES=0,1 python main.py --base-extractor $base_extractor --weights $weights \
--data $data --num-classes $n_classes --num-crops $num_crops --resize1 $resize1 --resize1-scale $resize1_scale \
--transform $transform --workers $workers --channels $channels --binarize --evaluate-by-case \
--self-attention  --drop-out $drop_out --model $model --model-dim $model_dim \
--model-dir $model_dir --head-dim $head_dim --in-dim $in_dim --n-layers $n_layers \
--linear-channel $linear_channel --num-scale-attn-layer $num_scale_attn_layer --weight-tie \
--aggregate-batch $aggregate_batch --batch-size $batch_size --use-gpu --gpu-id $gpu_id --use-parallel \
--use-standard-emb --warmup --loss-function $loss_function --lr $lr --lr-decay $lr_decay \
--patience $patience --max-bsz-cnn-gpu0 $max_bsz_cnn_gpu0 --mode $mode --optim $optim \
--savedir $savedir --scheduler $scheduler --weight-decay $weight_decay --epochs $epochs --start-epoch $start_epoch