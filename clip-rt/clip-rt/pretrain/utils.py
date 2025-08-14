"""
Code from OpenVLA (https://github.com/openvla/openvla)
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""
"""
data_utils.py

Additional RLDS-specific data utilities.
"""
import logging
import tensorflow as tf
import hashlib
import json
import os
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import dlimp as dl
import numpy as np
from tqdm import tqdm



# droid utils
def zero_action_filter(traj: Dict) -> bool:
    """
    Filters transitions whose actions are all-0 (only relative actions, no gripper action).
    Note: this filter is applied *after* action normalization, so need to compare to "normalized 0".
    """
    DROID_Q01 = tf.convert_to_tensor(
        [
            -0.7776297926902771,
            -0.5803514122962952,
            -0.5795090794563293,
            -0.6464047729969025,
            -0.7041108310222626,
            -0.8895104378461838,
        ]
    )
    DROID_Q99 = tf.convert_to_tensor(
        [
            0.7597932070493698,
            0.5726242214441299,
            0.7351000607013702,
            0.6705610305070877,
            0.6464948207139969,
            0.8897542208433151,
        ]
    )

    DROID_NORM_0_ACT = 2 * (tf.zeros_like(traj["action"][:, :, :6]) - DROID_Q01) / (DROID_Q99 - DROID_Q01 + 1e-8) - 1
    return tf.reduce_any(tf.math.abs(traj["action"][:, :, :6] - DROID_NORM_0_ACT) > 1e-5)

def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]

    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory

def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def tree_map(fn: Callable, tree: Dict) -> Dict:
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}


def tree_merge(*trees: Dict) -> Dict:
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    else:
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# === State / Action Processing Primitives ===


# ruff: noqa: B023
def normalize_action_and_proprio(traj: Dict, metadata: Dict, normalization_type: NormalizationType):
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""
    keys_to_normalize = {"action": "action", "proprio": "observation/proprio"}

    if normalization_type == NormalizationType.NORMAL:
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x),
            )

        return traj

    elif normalization_type in [NormalizationType.BOUNDS, NormalizationType.BOUNDS_Q99]:
        for key, traj_key in keys_to_normalize.items():
            if normalization_type == NormalizationType.BOUNDS:
                low = metadata[key]["min"]
                high = metadata[key]["max"]
            elif normalization_type == NormalizationType.BOUNDS_Q99:
                low = metadata[key]["q01"]
                high = metadata[key]["q99"]
            mask = metadata[key].get("mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool))
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1),
                    x,
                ),
            )

            # Note (Moo Jin): Map unused action dimensions (i.e., dimensions where min == max) to all 0s.
            zeros_mask = metadata[key]["min"] == metadata[key]["max"]
            traj = dl.transforms.selective_tree_map(
                traj, match=lambda k, _: k == traj_key, map_fn=lambda x: tf.where(zeros_mask, 0.0, x)
            )

        return traj

    raise ValueError(f"Unknown Normalization Type {normalization_type}")


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


# === RLDS Dataset Initialization Utilities ===
def pprint_data_mixture(dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]) -> None:
    print("\n######################################################################################")
    print(f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #")
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print("######################################################################################\n")


def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Either computes the statistics of a dataset or loads them from a cache file if this function has been called before
    with the same `hash_dependencies`.

    Currently, the statistics include the min/max/mean/std of the actions and proprio as well as the number of
    transitions and trajectories in the dataset.
    """
    unique_hash = hashlib.sha256("".join(hash_dependencies).encode("utf-8"), usedforsecurity=False).hexdigest()

    # Fallback local path for when data_dir is not writable or not provided
    local_path = os.path.expanduser(os.path.join("~", ".cache", "orca", f"dataset_statistics_{unique_hash}.json"))
    if save_dir is not None:
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json")
    else:
        path = local_path

    # check if cache file exists and load
    if tf.io.gfile.exists(path):
        logging.info(f"Loading existing dataset statistics from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            metadata = json.load(f)
        return metadata

    if os.path.exists(local_path):
        logging.info(f"Loading existing dataset statistics from {local_path}.")
        with open(local_path, "r") as f:
            metadata = json.load(f)
        return metadata

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": (
                traj["observation"]["proprio"] if "proprio" in traj["observation"] else tf.zeros_like(traj["action"])
            ),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    logging.info("Computing dataset statistics. This may take a bit, but should only need to happen once.")
    actions, proprios, num_transitions, num_trajectories = [], [], 0, 0
    for traj in tqdm(dataset.iterator(), total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1

    actions, proprios = np.concatenate(actions), np.concatenate(proprios)
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "q01": np.quantile(proprios, 0.01, axis=0).tolist(),
            "q99": np.quantile(proprios, 0.99, axis=0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        logging.warning(f"Could not write dataset statistics to {path}. Writing to {local_path} instead.")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(metadata, f)

    return metadata


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logging.info(f"Saved dataset statistics file at path {out_path}")


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """
    Allocates an integer number of threads across datasets based on weights.

    The final array sums to `n`, but each element is no less than 1. If `n` is None, then every dataset is assigned a
    value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # Give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)

        # Recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1

    return allocation



def bridge_oxe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def bridge_orig_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def ppgm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.reshape(eef_value, (-1, 7))
    gripper_value = tf.io.decode_compressed(trajectory["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1, 1))
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def taco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["robot_obs"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["robot_obs"][:, 7:8]
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["end_effector_cartesian_pos"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["end_effector_cartesian_pos"][:, -1:]

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_cable_routing_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][..., 0]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., -1:]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def kaist_nonprehensible_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["end_effector_pose"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 7:8]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:7]

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["gripper_state"]),
        ),
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"],
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_oxe": bridge_oxe_dataset_transform,
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    "ppgm": ppgm_dataset_transform,
    "ppgm_static": ppgm_dataset_transform,
    "ppgm_wrist": ppgm_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
    "droid": droid_baseact_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": tdroid_dataset_transform,
    "tdroid_pour_corn_in_pot": tdroid_dataset_transform,
    "tdroid_flip_pot_upright": tdroid_dataset_transform,
    "tdroid_move_object_onto_plate": tdroid_dataset_transform,
    "tdroid_knock_object_over": tdroid_dataset_transform,
    "tdroid_cover_object_with_towel": tdroid_dataset_transform,
    ### DROID Finetuning datasets
    "droid_wipe": droid_finetuning_transform,
}