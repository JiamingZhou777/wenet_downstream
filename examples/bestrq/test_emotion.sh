#!/bin/bash
export PYTHONPATH=/mnt/petrelfs/zhongguipin/git/wenet_downstream
export NCCL_DEBUG=INFO

exp_tag="100m_ln_23epoch"
echo "begin testing"
# export master=`scontrol show hostname $SLURM_NODELIST | head -n1`
# nodes=`scontrol show hostname $SLURM_NODELIST | wc -l`
# node_rank=`echo $SLURM_PROCID`
# master_addr=`python -c 'import socket; import os  ; print(socket.gethostbyname(os.environ["master"]))'` ;
# echo $ip
# echo master ${master}
# echo nodes  ${nodes}
# echo master_addr ${master_addr}
# echo node_rank ${node_rank}
export CUDA_VISIBLE_DEVICES="0"

test_path=$PYTHONPATH/wenet/bin/recognize_emotion.py

test_shards=data/fold1/test.list
model=exp/$exp_tag/ckpts/

average_checkpoint=true
average_num=5
# decode_checkpoint=${model}/390.pt
decode_checkpoint=${model}/avg_${average_num}.pt
decode_modes="emotion_recognition"


Test model, please specify the model you want to test by --checkpoint
if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$model/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python ../../wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $model  \
        --num ${average_num} \
        --val_best
fi

#   # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
decoding_chunk_size=
ctc_weight=0.5
reverse_weight=0.0

for mode in ${decode_modes}; do
{
test_dir=${model}/test_${mode}
mkdir -p $test_dir
   python ${test_path} --gpu 1 \
    --mode ${mode} \
    --config ./train_emotion.yaml \
    --data_type raw \
    --test_data ${test_shards} \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 1 \
    --penalty 0.0 \
    --dict data/dict \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_file $test_dir/text \
    --connect_symbol ' '  ## used for phone 
echo 'done'
} 
done
