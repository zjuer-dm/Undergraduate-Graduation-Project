import os
import random
import sys
from typing import List, Optional, Type, Union

import habitat
from habitat import logger
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset
from habitat_baselines.utils.env_utils import make_env_fn

random.seed(0)

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)


def is_slurm_job() -> bool:
    return SLURM_JOBID is not None


def is_slurm_batch_job() -> bool:
    r"""Heuristic to determine if a slurm job is a batch job or not. Batch jobs
    will have a job name that is not a shell unless the user specifically set the job
    name to that of a shell. Interactive jobs have a shell name as their job name.
    """
    return is_slurm_job() and os.environ.get("SLURM_JOB_NAME", None) not in (
        None,
        "bash",
        "zsh",
        "fish",
        "tcsh",
        "sh",
    )


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    # 这里的config.SIMULATOR_GPU_IDS已经变成了单个的了，因为在ss_trainer_ETP中执行了：self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
    # 因此底下的把episodes按照scenes进行均分也只是在单个进程/单个gpu的多个envs之间进行分配，每个gpu进程下的所有envs加起来都是完整的dataset，而不是像预训练那样，每个gpu内部是dataset的一部分
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE) # 输入值：VLN-CE-v1，返回值类型：habitat_extensions.task.VLNCEDatasetV1
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES # ['*']
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        # 对于eval和infer模式，这里得到的scenes已经是专属于这个gpu中所分配的episodes对应的全部scenes了(在VLNCEDatasetV1中通过config.TASK_CONFIG.DATASET.EPISODES_ALLOWED实现的筛选)
        # 具体是通过collect_val_traj或collect_infer_traj进行的episodes分配，以episodes_allowed的形式输入本函数
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    logger.info(f"SPLTI: {config.TASK_CONFIG.DATASET.SPLIT}, NUMBER OF SCENES: {len(scenes)}")

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            # 将scenes根据当前gpu中的envs进行平均分配，每个env分配到一定数量的scenes
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j # 每个gpu下每一个env是一个独立的id，实际上就是每一个env是一个独立的id（只有当前gpu）

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config) 
        
    is_debug = True if sys.gettrace() else False # False
    env_entry = habitat.ThreadedVectorEnv if is_debug else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn, # dataset在这里被真正的实例化了
        env_fn_args=tuple(zip(configs, env_classes)), 
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs


def construct_envs_auto_reset_false(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> VectorEnv:
    return construct_envs(config, env_class, auto_reset_done=False)

def construct_envs_for_rl(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_envs > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )
        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            proc_config.freeze()
            configs.append(proc_config)

    is_debug = True if sys.gettrace() else False
    env_entry = habitat.ThreadedVectorEnv if is_debug else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)),
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
