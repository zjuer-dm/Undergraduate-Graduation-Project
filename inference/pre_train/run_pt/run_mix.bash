
NODE_RANK=0
NUM_GPUS=4
outdir=pretrained/r2r_rxr_ce_30/mymlm.sap_habitat_depth

# train
# python -m torch.distributed.launch \
#     --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
#     pretrain_src/pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
#     --vlnbert cmt \
#     --model_config pretrain_src/run_pt/r2r_model_config_dep.json \
#     --config pretrain_src/run_pt/r2r_pretrain_habitat.json \
#     --output_dir $outdir
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
    pretrain_src/my_pretrain_src_30/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/mix_model_config_dep.json \
    --config pretrain_src/run_pt/mix_pretrain_habitat_my_pretrain_30.json \
    --output_dir $outdir


# CUDA_VISIBLE_DEVICES=0 bash pretrain_src/run_pt/run_mix.bash 2333


# A6000服务器上先在终端运行下面这条
# export NCCL_P2P_DISABLE=1
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix.bash 2333
