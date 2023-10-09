#!/bin/bash
export PYTHONPATH=/mnt/petrelfs/zhongguipin/git/wenet_downstream
export NCCL_DEBUG=INFO

exp_tag="100m_ln_23epoch"
echo "begin training"
export master=`scontrol show hostname $SLURM_NODELIST | head -n1`
nodes=`scontrol show hostname $SLURM_NODELIST | wc -l`
node_rank=`echo $SLURM_PROCID`
master_addr=`python -c 'import socket; import os  ; print(socket.gethostbyname(os.environ["master"]))'` ;
echo $ip
echo master ${master}
echo nodes  ${nodes}
echo master_addr ${master_addr}
echo node_rank ${node_rank}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

train_path=$PYTHONPATH/wenet/bin/train_spring_emotion.py
train_shards=data/fold1/train.list
cv_shards=data/fold1/test.list

model=exp/$exp_tag/ckpts/
tensorboard=exp/$exp_tag

config=./train_emotion.yaml

mkdir -p $model
mkdir -p $tensorboard
cmvn=
checkpoint= #exp/$exp_tag/final.pt

# data type raw
time python launch.py  --nproc_per_node=8 --master_port=52071 \
           --nnodes=$nodes \
           --master_addr $master_addr \
           --node_rank=$node_rank \
           ${train_path} --gpu 1 \
           --data_type raw \
           --train_data  ${train_shards} \
           --model_dir ${model} \
           --symbol_table data/dict \
           --ddp.init_method "env://" \
           --cv_data ${cv_shards} \
           --config ${config} \
           --num_workers 4 \
           --tensorboard_dir  ${tensorboard} 
           #--checkpoint $checkpoint