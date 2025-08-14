"""
CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision
https://arxiv.org/abs/2411.00508

We modify the data curation code of OpenVLA (https://github.com/openvla/openvla).
The goal of this code is to convert low-level end-effector actions in Open X-Embodiment (OXE) data into natural language supervision.

The transformation is divided into three steps:

1. load the OXE data
2. identify the dominant axis in 7-dimensional end-effector actions and correponding values
3. mapping such information to one of predefined natural language supervisions (in nls.py)

However, each dataset in OXE uses different 

1. units (some used radians and meter, the others used degree and centimeters) 
2. definition of Cartesian coordinates.

So, before transforming the end-effector action into natural language supervision, we standardize units and xyz coordinates (see adjust_actions_scale_dimensions).

The preprocessed data is the format of *.tar file (webdataset)
"""

import os
import io
import json
import torch
import random
import copy
import webdataset as wds
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import IterableDataset
from nls import NATURAL_LANGUAGE_SUPERVISION, EEF_POS_THRESHOLDS, EEF_POS_DISCRETE_ACTIONS, EEF_ORI_THRESHOLDS, EEF_ORI_DISCRETE_ACTIONS
from configs import OXE_DATASET_CONFIGS, ActionEncoding
from utils import OXE_STANDARDIZATION_TRANSFORMS, NormalizationType
from dataset import make_single_dataset, make_interleaved_dataset


def load_per_dataset_kwargs(mixture_spec, data_root_dir, load_camera_views = ("primary",)):
    dataset_kwargs_list, sampling_weights = [], []
    
    # make per_dataset_arguments 
    for d_name, d_weight in mixture_spec:
        dataset_kwargs = copy.deepcopy(OXE_DATASET_CONFIGS[d_name])

        # use primary observation
        dataset_kwargs["image_obs_keys"] = {
            k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
        }
        dataset_kwargs["language_key"] = "language_instruction"
        dataset_kwargs['standardize_fn'] = OXE_STANDARDIZATION_TRANSFORMS[d_name]
        dataset_kwargs['shuffle'] = False

        # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
        # Normalize all action dimensions *except* the gripper
        if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
            dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
            dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
        elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
            dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
            dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
        dataset_kwargs["action_proprio_normalization_type"] = NormalizationType.NORMAL

        # Add any aux arguments
        if "aux_kwargs" in dataset_kwargs:
            dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

        # eliminate unnecessary keys
        dataset_kwargs.pop("state_encoding")
        dataset_kwargs.pop("action_encoding")
        dataset_kwargs.pop("depth_obs_keys")
        dataset_kwargs.pop("state_obs_keys")

        # append per_dataset_arguments

        dataset_kwargs_list.append(
            {
                "name": d_name, 
                "data_dir": str(data_root_dir), 
                **dataset_kwargs
            }
        )
        sampling_weights.append(d_weight)
    return dataset_kwargs_list, sampling_weights


def adjust_actions_scale_dimensions(dataset_name, actions, images, instructions):
    """
    Each dataset has different definition regarding (x,y,z) coordinates and units for position and rotation 
    So, we standardize the 3d coordinates and units as:
    3d coordinates:
        The x-axis is in the depth direction, increasing away from the base
        The y-axis is in the horizontal direction, increasing to the left
        The z-axis is in the vertical direction, increasing upwards.

    Units:
        Position uses meter as the basic unit
        Rotation uses radian as the basic unit
    """
    if dataset_name == 'fractal20220817_data':
        images = images[1:]
        actions = actions[:-1]
        instructions = instructions[:-1]
    elif dataset_name == 'kuka':
        if instructions.shape[0] > 8:
            images = images[1:]
            actions = actions[:-1]
            instructions = instructions[:-1]
    elif dataset_name == 'bridge_orig':
        pass
    elif dataset_name == 'taco_play':
        actions = position_meter(actions)
        actions = rotation_radian(actions)
        actions[:, :, 0] *= -1
    elif dataset_name == 'jaco_play':
        actions = position_meter(actions)
        actions = swap_x_y(actions)
        actions[:, :, 1] *= -1 
    elif dataset_name == 'berkeley_cable_routing':
        actions = position_meter(actions)
        actions = swap_x_y(actions)
        actions[:, :, 1] *= -1
        actions[:, :, 2] *= -1
    elif dataset_name == 'roboturk':
        actions[:, :, 0] *= -1
        actions[:, :, 1] *= -1
    elif dataset_name == 'viola':
        actions = position_meter(actions)
        actions = rotation_radian(actions)
        actions = swap_x_y(actions)
        actions[:, :, 1] *= -1
    elif dataset_name == 'berkeley_autolab_ur5':
        actions = swap_x_y(actions)
        actions[:, :, 2] *= -1
    elif dataset_name == 'toto':
        actions = position_meter(actions)
        actions = rotation_radian(actions)
    elif dataset_name == 'language_table':
        actions[:, :, 0] *= -1
        actions[:, :, 1] *= -1
    elif dataset_name == 'stanford_hydra_dataset_converted_externally_to_rlds':
        actions[:, :, 0] *= -1
        actions[:, :, 1] *= -1    
    elif dataset_name == 'austin_buds_dataset_converted_externally_to_rlds':
        actions = position_meter(actions)
        actions[:, :, 0] *= -1
        actions[:, :, 1] *= -1
    elif dataset_name == 'nyu_franka_play_dataset_converted_externally_to_rlds':
        pass
    elif dataset_name == 'furniture_bench_dataset_converted_externally_to_rlds':
        actions[:, :, 0] *= -1
        actions[:, :, 1] *= -1
    elif dataset_name == 'ucsd_kitchen_dataset_converted_externally_to_rlds':
        actions = np.diff(actions, axis=0)
        actions[:, :, :3] /= 1000
        actions = rotation_radian(actions)
        actions = swap_x_y(actions)
        actions[:, :, 1] *= -1
        images = images[:-1]
        instructions = instructions[:-1]
    elif dataset_name == 'austin_sailor_dataset_converted_externally_to_rlds':
        actions = position_meter(actions)
        actions = rotation_radian(actions)
        actions = swap_x_y(actions)
        actions[:, :, 1] *= -1
    elif dataset_name == 'austin_sirius_dataset_converted_externally_to_rlds':
        actions = position_meter(actions)
        actions = rotation_radian(actions)
        actions[:, :, 0] *= -1
        actions[:, :, 1] *= -1    
    elif dataset_name == 'dlr_edan_shared_control_converted_externally_to_rlds':
        actions[:, :, 0] *= -1    
    elif dataset_name == 'iamlab_cmu_pickup_insert_converted_externally_to_rlds':
        actions = np.diff(actions, axis=0)
        actions = swap_x_y(actions)
        actions[:, :, 1] *= -1
        images = images[:-1]
        instructions = instructions[:-1]
    elif dataset_name == 'utaustin_mutex':
        actions = position_meter(actions)
        actions = rotation_radian(actions)    
    elif dataset_name == 'cmu_stretch':
        pass
    elif dataset_name == 'bc_z':
        pass
    elif dataset_name == 'fmb_dataset':
        actions = position_meter(actions)
        actions = rotation_radian(actions)
        actions = swap_x_y(actions)
        actions[:, :, 1] *= -1
    else:
        raise NotImplementedError("should implement")
    return actions, images, instructions

def position_meter(actions):
    actions[:, :, :3] /= 100
    return actions

def rotation_radian(actions):
    return np.concatenate((actions[:, :, :3], np.radians(actions[:, :, 3:6])), axis=-1)

def swap_x_y(actions):
    actions.T[[0, 1]] = actions.T[[1, 0]]
    return actions

def dominant_axis_and_values(action, scale_pos_ori=7.854):
    """
    Determine the dominant axis for each Cartesian end-effector command and return the values.

    Parameters:
    action: the unnormalized delta end-effector action(6) and absolute gripper action(1).
    Returns:
    np.ndarray: An array containing the dominant axis for each action data point.
    """
    action = np.squeeze(action, axis=1) # (b, 7)

    eef_action = action[:, :-1]
    grp_action = action[:, -1]

    # compute the absolute values for end-effector action
    abs_eef_action = np.abs(eef_action)

    # scale position (meter) and orientation (radian) values
    # we multiply position values by 7.854 (0.1 meter and 0.7854 radian) to balance between the two
    abs_eef_action[:, :3] *= scale_pos_ori

    # determine the dominant axis for synthesizing natural language supervision
    dominant_eef_axis = np.argmax(abs_eef_action, axis=1)

    # Get the original value
    dominant_eef_val = eef_action[np.arange(eef_action.shape[0]), dominant_eef_axis]
    return dominant_eef_axis, dominant_eef_val, grp_action


def discretize_eef_val(eef_val, axis):
    discretized_action = [0.0] * 6

    if axis <= 2:
        bin_idx = np.digitize(eef_val, EEF_POS_THRESHOLDS, right=True)
        discretized_val = EEF_POS_DISCRETE_ACTIONS[bin_idx]
    else:
        bin_idx = np.digitize(eef_val, EEF_ORI_THRESHOLDS, right=True)
        discretized_val = EEF_ORI_DISCRETE_ACTIONS[bin_idx]

    discretized_action[axis] = discretized_val
    return discretized_action, bin_idx


    
def map_language_supervision(axis, eef_val, grp_val, eef_tau=[0.01, 0.08727], grp_tau=0.5):
    """
    Convert the dominant action axis into natural language supervision

    axis : dominant indices ranging from 0 to 5
    eef_val: dominant values
    grp_val: gripper value (mostly 0 or 1)
    eef_tau: the threshold value for position (0.01m) and orientation 0.08727 rad (5 degrees)
    """

    # Check each absolute value for end-effector pose exceeds the minimum threshold value to make natural language supervision
    ori_or_pos = (axis > 2).astype(int) # 1 if the dominant axis is orientation, else 0 (position)
    eef_tau = np.array(eef_tau)

    # Check the dominant axis exceed the minimum threshold 
    eef_flags = np.where(abs(eef_val) > eef_tau[ori_or_pos], True, False)
    
    """
    Gripper open / close state has absolute values
    So, we compute relative gripper values by computing the difference between two consecutive time steps
    Finally, we check the delta gripper state transition exceeds the threshold value 0.5
    """
    grp_delta = np.diff(grp_val)
    grp_delta = np.append(grp_delta, 0)
    grp_flags = np.where(abs(grp_delta) > grp_tau, True, False)

    lang_sups = []
    disc_actions = []

    # Based on information above, we synthesize natural language supervision
    for idx, (eef_flag, grp_flag) in enumerate(zip(eef_flags, grp_flags)):
        eef_sup, grp_sup = None, None
        
        # Load language supervision for end-effector
        if eef_flag:
            """
            We discretize the end-effector position / rotation values into one of pre-defined granularities.
            Position:
                1cm, 5cm, 10cm, 20cm
            Rotation:
                5 degrees, 15 degrees, 45 degrees, 90 degrees

            After mapping to discrete action values, we synthesize the action to natural language supervision
            """
            disc_eef_action, bin_idx = discretize_eef_val(eef_val[idx], axis[idx])    
            eef_sup_key = str(axis[idx]) + "_" + str(bin_idx)
            eef_sup = random.choice(NATURAL_LANGUAGE_SUPERVISION["end_effector"][eef_sup_key])

        # Load language supervision for gripper
        if grp_flag:
            # if the delta value is greater than 0, it implies opening the gripper 
            if grp_delta[idx] > 0:
                disc_grp_action = [1.0]
                grp_sup = NATURAL_LANGUAGE_SUPERVISION["gripper"]["0_1"]
            else:
                disc_grp_action = [0.0] 
                grp_sup = NATURAL_LANGUAGE_SUPERVISION["gripper"]["0_0"]

        """
        Language supervision is either end-effector displacement or gripper open/close transition
        If all end-effector and gripper values are below threshold, language supervision set to none 
        """ 
        if grp_sup:
            lang_sup = grp_sup
            disc_eef_action = [0.0] * 6
        elif eef_sup:
            lang_sup = eef_sup
            disc_grp_action = [1.0]
        else:
            lang_sup = None
            disc_eef_action = [0.0] * 6
            disc_grp_action = [1.0]

        disc_action = disc_eef_action + disc_grp_action 
        lang_sups.append(lang_sup)
        disc_actions.append(disc_action)
    return lang_sups, disc_actions


def array_to_bytes(image_array, format= 'JPEG'):
    image = Image.fromarray(image_array).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return image_bytes


def unnorm_action(action, mean, std):
    eef_action = action[:, :, :-1]
    grp_action = action[:, :, -1:]

    # unnormalize actions using mean and std
    mean = np.array(mean[:-1])
    std = np.array(std[:-1])
    unnormalized_eef_action = eef_action * (std + 1e-8) + mean
    return np.concatenate((unnormalized_eef_action, grp_action), axis=-1)


if __name__ == '__main__':

    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)
    tf.random.set_seed(7)

    # The list of datasets for preprocessing
    mixture_spec = [
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
        ("kuka", 0.8341046294),
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        #("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ## New Datasets in MagicSoup++
        ("bc_z", 0.2),                                          # Note: use v0.1.0 --> later versions broken
        ("fmb_dataset", 1.0),
        ("dobbe", 0.2),
        ("droid", 0.06),
    ]

    # This denotes the path for open X-embodiment dataset curated by OpenVLA
    raw_data_root = '/data/OXE_download'
    dataset_kwargs_list, sampling_weights = load_per_dataset_kwargs(mixture_spec, raw_data_root)


    rlds_config = dict(
        traj_transform_kwargs=dict(
            window_size=1,                                      # If we wanted to feed / predict more than one step
            future_action_window_size=0,                        # For action chunking
            skip_unlabeled=True,                                # Skip trajectories without language labels
            goal_relabeling_strategy="uniform",                 # Goals are currently unused
        ),
        frame_transform_kwargs=dict(
            resize_size=(224, 224),                             # Resolution for resizing 
            num_parallel_calls=16,                              # For CPU-intensive ops (decoding, resizing, etc.)
        ),
        train=True,
    )


    # The path for saving preprocessed data
    data_root = '/data/oxe'
    action_to_token = json.load(open('../docs/action_to_label_pretrain.json', 'r'))

    for dataset_kwargs in dataset_kwargs_list:
        # load each dataset and get the statistics of the data
        rlds_config["dataset_kwargs"] = dataset_kwargs
        dataset, dataset_length, dataset_statistics = make_single_dataset(**rlds_config)
        mean, std = dataset_statistics['action']['mean'], dataset_statistics['action']['std']
        num_trajectories = dataset_statistics['num_trajectories']

        # generate a directory for image
        dataset_name = dataset_kwargs['name']
        save_data_path = os.path.join(data_root, dataset_name)
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path, exist_ok=True)
        print(dataset_name + ' processing doing ...')

        idx = 0   
        # We save shards for each dataset
        with wds.ShardWriter(os.path.join(save_data_path, '%06d.tar'), maxcount=1000) as sink: 
            for rlds_batch in dataset.as_numpy_iterator():
                actions = rlds_batch["action"]                             # (b, 1, 7) 
                images = rlds_batch["observation"]["image_primary"]        # (b, 1, 224, 224, 3)   
                instructions = rlds_batch["task"]["language_instruction"]  # (b,)

                # check all batch samples are in the same episode
                if len(list(set(instructions))) != 1:
                    print('inconsistent batch occurs')
                    continue

                # unnormalize low-level action using data statistics
                actions = unnorm_action(actions, mean, std) 

                # standardize 3D Cartesian coordinates and units for position and roatation
                actions, images, instructions = adjust_actions_scale_dimensions(dataset_name, actions, images, instructions)
                
                # find the dominant axis in low-level action
                axis, eef_val, grp_val = dominant_axis_and_values(actions)

                # map the axis to natural language supervision
                lang_sup, disc_actions = map_language_supervision(axis, eef_val, grp_val)

                assert images.shape[0] == instructions.shape[0] == len(lang_sup) == len(disc_actions)
                batch_size = images.shape[0] 

                for j in range(batch_size):
                    supervision = lang_sup[j]
                    # skip samples whose language supervision is None
                    if supervision:
                        image = array_to_bytes(images[j][0])
                        instruction = "what motion should the robot arm perform to complete the instruction '{}'?".format(instructions[j].lower().decode())
                        cls_action = action_to_token[str(disc_actions[j])] # generate a class label for each language supervision

                        sink.write(
                            {
                                '__key__': "%06d" % idx,
                                'jpg': image,
                                'txt': instruction,
                                'sup': supervision,
                                'cls': cls_action
                            }
                        )
                    idx += 1

