import os
import h5py
import json
import numpy as np
import glob
import cv2
import argparse
from tqdm import tqdm
from PIL import Image

def process_libero_to_cliprt(hdf5_file, output_base_dir):
    file_name = os.path.basename(hdf5_file).replace("_demo.hdf5", "")
    instruction = file_name.replace("_", " ")
    output_dir = os.path.join(output_base_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_file, "r") as f:
        if "data" not in f:
            print(f"Error: 'data' group doesn't exist. ({hdf5_file})")
            return
        
        data_group = f["data"]
        for demo in tqdm(sorted(data_group.keys()), desc=f"Processing {file_name}", leave=False):
            demo_data = data_group[demo]
            actions = np.array(demo_data["actions"])
            joint_states = np.array(demo_data["obs/joint_states"])
            robot_states = np.array(demo_data["robot_states"])
            agentview_rgb = np.array(demo_data["obs/agentview_rgb"])

            demo_output_dir = os.path.join(output_dir, demo)
            os.makedirs(demo_output_dir, exist_ok=True)
            
            for i in tqdm(range(len(actions)), desc=f"Processing {demo}", leave=False):
                image_filename = f"{demo}_timestep_{i:04d}.png"
                image_path = os.path.join(demo_output_dir, image_filename)
                Image.fromarray(agentview_rgb[i]).save(image_path)

                timestep_data = {
                    "joint": joint_states[i].tolist(),
                    "instruction": instruction,
                    "action": actions[i].tolist(),
                    "image_path": image_filename,
                    "states": robot_states[i].tolist(),
                }
                
                json_path = os.path.join(demo_output_dir, f"{demo}_timestep_{i:04d}.json")
                with open(json_path, "w") as json_file:
                    json.dump(timestep_data, json_file, indent=4)
    
        print(f"{file_name}: All timestep data saved successfully!")


def process_all_hdf5(input_dir, output_base_dir):
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    print(f"📂 Found {len(hdf5_files)} files. Processing...")
    for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 Files"):
        process_libero_to_cliprt(hdf5_file, output_base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Libero HDF5 to Raw ")
    parser.add_argument("--source", "--dataset_path", required=True, help="Path to the dataset directory")
    parser.add_argument("--target", "--output_path", required=True, help="Path to save the converted dataset")
    
    args = parser.parse_args()
    
    input_directory = args.dataset_path
    output_directory = args.output_path
    
    print(f"Input Directory: {input_directory}")
    print(f"Output Directory: {output_directory}")
    
    process_all_hdf5(input_directory, output_directory)
