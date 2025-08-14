# Pretraining CLIP-RT from Scratch


To pretrain CLIP-RT on [Open X-Embodiment (OXE) dataset](https://robotics-transformer-x.github.io), we should preprocess the OXE data first and train it on the preprocessed data. The goal of preprocessing is to extract natural language supervision from low-level end-effector commands in OXE data.


## Preprocessing Data

We use the OXE dataset curated by [OpenVLA](https://github.com/openvla/openvla).

1. Follow the data installation guide in [OpenVLA](https://github.com/openvla/openvla). The [script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) will help download OXE data.  

2. Run `python preprocess.py`



## How to Extract Natural Language Supervision

We extract natural language supervision from 7D end-effector commands (the first six for position and rotation / the last one is gripper open and close). Specifically, we find the dominant axis in the end-effector commands and map it to natural language supervision. For example, the dominant axis is the z-axis with a value of -0.2, it is converted into: "lower the arm by twenty centimeters".

One problem is that each robotic dataset has different definitions about the Cartesian coordinate system and units for each value (e.g., meters, degrees, radians, etc). So, we manually check each robotic data in the OXE dataset and standardize those definitions (see preprocess.py).

Technically speaking, there can be an infinite number of natural language supervisions (e.g., "lower the arm by 11.52 centimeters"). However, to make VLA models more generalizable action representations, we predefine the granularities of robotic actions. Specifically, each axis for position control is divided into 8 granularities (-20cm, -10cm, -5cm, -1cm, 1cm, 5cm, 10cm, 20cm). Rotational control has 8 granularities (-90°, -45°, -15°, -5°, 5°, 15°, 45°, 90°). This means that the dominant axis values are discretized into one of those values. Finally, the discretized values are mapped into natural language supervisions. There are 50 types of natural language supervisions (8 * 6 dimensions, gripper open, and gripper close commands). See `docs/action_to_label_pretrain.json` for more details. 

We prepare 899 variants of natural langauge supervisions to make VLA models learn diverse natural language supervision. We use GPT-4 to prepare those supervisions.


## Pretrain CLIP-RT via Contrastaive Imitation Learning

After preprocessing OXE data, simply run `./scripts/pretrain.sh`.

We use 4 NVIDIA H100 GPUs for one day with a batch size of 128.
