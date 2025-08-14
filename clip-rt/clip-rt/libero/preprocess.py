import os
import io
import glob
import json
import random
import copy
import webdataset as wds
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse


def image_to_bytes(image, format='PNG'):
    # rotate image 180 degrees since the original image is upside down
    image = Image.open(image).convert("RGB").rotate(180)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    return image_bytes


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Preprocess Raw Data to Tar files")
    parser.add_argument("--source", "--dataset_path", required=True, help="Path to the root directory of raw data")
    parser.add_argument("--target", "--output_path", required=True, help="Path to save the converted dataset")
    parser.add_argument("--dname", "--dataset_name", required=True, choices=['goal', 'object', '10', 'spatial'], help="The name of LIBERO task suite")
    
    args = parser.parse_args()
    
    raw_data_root = args.dataset_path
    dataset_name = args.dataset_name
    save_data_path = args.output_path
 
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path, exist_ok=True)
    print(dataset_name + ' processing doing ...')

    # the dimension of each action is 7, and the size of action chunk is 8
    idx = 0
    action_dim = 7
    window_size = 8

    # save shards for each dataset
    with wds.ShardWriter(os.path.join(save_data_path, '%06d.tar'), maxcount=1000) as sink:
        for subdir in os.listdir(raw_data_root):
            s_dir = os.path.join(raw_data_root, subdir)
            print(s_dir)
            for subsubdir in os.listdir(s_dir):
                ss_dir = os.path.join(s_dir, subsubdir)
                json_list = sorted(glob.glob(ss_dir + "/*.json"))
                image_list = sorted(glob.glob(ss_dir + "/*.png"))

                actions = []
                instructions = []
                
                for idx, json_data in enumerate(json_list):
                    with open(json_data, "r") as f:
                        d = json.load(f)
                        instruction = d['instruction']
                        action = d['action']
                    instruction = "what motion should the robot arm perform to complete the instruction '{}'?".format(instruction.lower())
                    assert image_list[idx].replace(".png", ".json") == json_list[idx]
                    actions.append(action)
                    instructions.append(instruction)

                assert len(actions) == len(instructions) == len(image_list)
                batch_size = len(actions)

                # We concatenated 8 actions whose each dimension is 7, resulting in the vector with 56 entries
                # If the size of remaining actions is less than 8, we pad these entries with the value of 1.1 
                # The padded values are masked out during training
                for j in range(batch_size):
                    chunk = actions[j:j+window_size]
                    lc = len(chunk)
                    if lc < window_size:
                        dummy = [1.1] * action_dim
                        zeropad = [dummy] * (window_size - lc)
                        chunk.extend(zeropad)
                    chunk = np.array(chunk)
                    assert chunk.shape == (window_size, action_dim)

                    instruction = instructions[j]
                    image = image_to_bytes(image_list[j])
                    sink.write(
                    	{
                            '__key__': "%06d" % idx,
                            'jpg': image,
                            'txt': instruction,
                            'npy': chunk
                        }
                    )
                    idx += 1
