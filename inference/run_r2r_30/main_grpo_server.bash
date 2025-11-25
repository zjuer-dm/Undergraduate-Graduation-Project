export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# 代码运行架构：
# 节点：指的是一台独立的物理计算机或服务器，内部可能包含多个gpu。若节点为1则是单机多卡训练，若为1以上，则是多机多卡训练
# 进程(即nproc_per_node的值)：一个“进程”是一个正在运行的程序实例。在 PyTorch 分布式训练（尤其是使用 DistributedDataParallel）中，通常每个 GPU 会由一个独立的 Python 进程来控制和运行训练代码。
# local_rank：当前节点（电脑）下的所有进程（gpu）编号
# world_size/rank：整个分布式训练任务中，参与训练的全局总进程（gpu）数量/编号。对于单机多卡而言，rank就等于local_rank
# 全部代码都运行在一个master节点上；在节点中含有多个进程，每个进程使用一个单独的gpu；
# 每个进程中创建了多个habitat的envs，每个habitat的env包含多个scenes，负责运行该gpu需要运行的episodes中对应scenes的子集
# （torch.distributed.launch默认只能处理一个进程一张gpu的问题,因为训练需要保证每个gpu都至少能单独运行全部可训练权重的网络）


# NUM_ENVIRONMENTS：num_envs_per_gpu
# 只有当IL.load_from_ckpt为True时,IL.is_requeue为True才会起作用;而后者的作用是,改写IL.ckpt_to_load,将其设置为权重文件夹下iteration次数最多的那个权重


flag1="--exp_name release_r2r_30_grpo_seeker
      --run-type train
      --exp-config run_r2r_30/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 16
      ONLY_LAST_SAVEALL True
      TRAINER_NAME GRPO-30-no-drop-env-efficient
      GRPO_ORM.iters 500
      GRPO_ORM.lr 2e-5
      GRPO_ORM.warmup_iters 0
      GRPO_ORM.min_lr_ratio 0.25
      GRPO_ORM.log_every 10
      GRPO_ORM.load_from_ckpt True
      GRPO_ORM.ckpt_to_load data/logs/checkpoints/release_r2r_30_seeker_dino/ckpt.iter200.pth
      GRPO_ORM.is_requeue False
      GRPO_ORM.waypoint_aug  True
      GRPO_ORM.sample_num 8
      GRPO_ORM.update_epochs 1
      GRPO_ORM.grpo_beta 0.04
      GRPO_ORM.grpo_epsilon 0.2
      GRPO_ORM.enable_amp False
      GRPO_ORM.enable_all_dropouts True
      GRPO_ORM.dropout_in_sampling True
      GRPO_ORM.dropout_rate 0.10
      GRPO_ORM.max_grad_norm 2.0
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      TASK_CONFIG.DATASET.SUFFIX _10
      MODEL.pretrained_path pretrained/r2r_rxr_ce_30/mlm.sap_habitat_depth_seeker_dino/ckpts/model_step_15000.pt
      "


mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag1
      ;;
esac

# 命令行运行：
# CUDA_VISIBLE_DEVICES=1,2 bash run_r2r_30/main_grpo_server.bash train 2333