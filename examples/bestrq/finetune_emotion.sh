#!/bin/bash
export PYTHONPATH=/mnt/petrelfs/zhongguipin/git/wenet_downstream
export NCCL_DEBUG=INFO

exp_tag="20230919_160000pt"
#cmvn=/mnt/petrelfs/zhongguipin/git/wenet_downstream/examples/bestrq/exp/20230905_180000pt/librispeech_global_cmvn
#20230804 except last encoder layer

echo "begin training"
# export CUDA_VISIBLE_DEVICES="0"
echo $CUDA_VISIBLE_DEVICES

train_path=$PYTHONPATH/wenet/bin/train_spring_emotion.py
train_shards=data/fold1/train.list
cv_shards=data/fold1/test.list

model=exp/$exp_tag/ckpts/
#tensorboard=exp/$exp_tag/tensorboard/finetune_bz1_nosort_2optimizer
tensorboard=exp/$exp_tag/tensorboard/test_bz8_cmvn_1e-5_s3prl_2linear
# config=./train_emotion.yaml
# config=./conf/train_conformer.yaml
config=./exp/$exp_tag/bestrq.yaml

mkdir -p $model
mkdir -p $tensorboard
cmvn=exp/$exp_tag/librispeech_global_cmvn
# checkpoint=exp/20230804/97000.pt
checkpoint=exp/$exp_tag/160000.pt

# b /mnt/petrelfs/zhongguipin/git/wenet_downstream/wenet/transformer/emotion_recognition_model.py
python   ${train_path} --gpu "0" \
           --data_type raw \
           --train_data  ${train_shards} \
           --model_dir ${model} \
           --symbol_table data/dict \
           --ddp.init_method "env://" \
           --cv_data ${cv_shards} \
           --config ${config} \
           --num_workers 4 \
           --tensorboard_dir  ${tensorboard} \
           --checkpoint $checkpoint  \
           --cmvn  $cmvn 

# num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
# num_nodes=1
# node_rank=0
# world_size=`expr $num_gpus \* $num_nodes`

# for ((i = 0; i < $num_gpus; ++i)); do
#     {
#       gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
#       # Rank of each gpu/process used for knowing whether it is
#       # the master of a worker.
#       rank=`expr $node_rank \* $num_gpus + $i`
#       dist_backend="nccl"
#       python ${train_path} --gpu $gpu_id \
#            --ddp.world_size $world_size \
#            --ddp.rank $rank \
#            --ddp.dist_backend $dist_backend \
#            --data_type raw \
#            --train_data  ${train_shards} \
#            --model_dir ${model} \
#            --symbol_table data/dict \
#            --ddp.init_method "env://" \
#            --cv_data ${cv_shards} \
#            --config ${config} \
#            --num_workers 4 \
#            --tensorboard_dir  ${tensorboard} \
#            --checkpoint $checkpoint 
#     } &
#     done

