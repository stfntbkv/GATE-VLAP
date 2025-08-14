"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # CLIP-RT:
    python run_libero_eval_clip_rt_reg.py \
        --model_family clip_rt \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ]
"""

import draccus
import numpy as np
import tqdm
import sys
import os
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

sys.path.append("./LIBERO")

from libero.libero import benchmark

from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)

from clip_rt_utils import get_clip_rt, get_tokenizer
from robot_utils import (
    DATE_TIME,
    set_seed_everywhere,
    get_clip_rt_action,
)

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import logging
logging.basicConfig(level=logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


@dataclass
class GenerateConfig:
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "clip_rt"  # Model family
    pretrained_checkpoint: Union[str, Path] = ""  # Pretrained checkpoint path

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # CLIP-RT environment-specific parameters
    #################################################################################################################
    save_video: str = "n"
    model_ckpt: str = "1"
    chunk_cut: int = 8
    save_action: str = 'n'
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None  # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"  # Local directory for eval logs

    seed: int = 7  # Random Seed (for reproducibility)


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    model_path = cfg.pretrained_checkpoint

    lines = []
    lines.append(str(cfg.model_family))
    lines.append(f"model epoch: {cfg.model_ckpt}")
    lines.append(f"Model path: {model_path}")
    lines.append(f"model epoch_{cfg.model_ckpt}")
    lines.append(f"chunk cut: {cfg.chunk_cut}")
    print("\n".join(lines))

    assert (
        cfg.pretrained_checkpoint is not None
    ), "cfg.pretrained_checkpoint must not be None!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA Convention] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    if cfg.model_family == "clip_rt":
        model, preprocess = get_clip_rt(
            model_path=model_path,
            task_split=cfg.task_suite_name,
        )
        tokenizer = get_tokenizer()

    # Initialize local logging
    run_id = f"EVAL-{cfg.model_family}-{cfg.task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    log_dir = os.path.join(cfg.local_log_dir, cfg.task_suite_name)
    os.makedirs(log_dir, exist_ok=True)
    local_log_filepath = os.path.join(log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = 224

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        print(f"Task {task_id} of {num_tasks_in_suite}")
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        tot_inf_time = 0
        tot_steps = 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            actions = []

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action(cfg.model_family)
                        )
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    # replay_images.append(img)

                    # Prepare observations dict
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                    }

                    stime = time.time()

                    action_chunks = get_clip_rt_action(
                        model,
                        preprocess,
                        tokenizer,
                        observation,
                        task_description,
                    )
                    etime = time.time()
                    runtime = etime - stime
                    tot_inf_time += runtime
                    tot_steps += 1

                    # print(f"Action_chunks: {action_chunks}")

                    log_file.write(f"Action_chunks: {action_chunks}\n")
                    actions.extend(action_chunks)

                    done_flag = False

                    # Execute action in environment
                    for _, action_chunk in enumerate(action_chunks):
                        # Get preprocessed image
                        img = get_libero_image(obs, resize_size)

                        # Save preprocessed image for replay video
                        replay_images.append(img)
                        obs, _, done, _ = env.step(action_chunk)
                        if done:
                            done_flag = True
                            break
                        t += 1

                    if done_flag:
                        task_successes += 1
                        total_successes += 1
                        break

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    # raise e
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if cfg.save_video == "y":
                save_rollout_video(
                    cfg.task_suite_name,
                    cfg.model_ckpt,
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                )
            if cfg.model_family == "clip_rt":
                if cfg.save_action == 'y':
                    import json
                    os.makedirs(
                        f"./experiments/actions/{cfg.task_suite_name}/epoch_{cfg.model_ckpt}/",
                        exist_ok=True,
                    )

                    with open(
                        f"./experiments/actions/{cfg.task_suite_name}/epoch_{cfg.model_ckpt}/actions_{task_description}_{episode_idx}.json",
                        "w",
                    ) as f:
                        json.dump(actions, f, indent=4)

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        # Log final results
        print("latency (sec): {}".format(float(tot_inf_time) / int(tot_steps)))

        print(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        print(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )
        log_file.write(
            "latency (sec): {}\n".format(float(tot_inf_time) / int(tot_steps))
        )
        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        log_file.flush()

    print("\n".join(lines))
    log_file.write("\n".join(lines))
    log_file.flush()
    # Save local log file
    log_file.close()


if __name__ == "__main__":
    eval_libero()
