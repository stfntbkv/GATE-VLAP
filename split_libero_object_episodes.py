#!/usr/bin/env python3
"""
Split LIBERO episodes into primitive action tasks with correct gripper detection.
Creates one HDF5 file per primitive action task, containing all demos for that task.
"""

import os
import h5py
import numpy as np
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import glob
import re
from collections import defaultdict


@dataclass
class ActionSegment:
    """Represents a segment of an episode corresponding to a primitive action."""
    action_type: str  # grasp, place, putin
    start_idx: int
    end_idx: int
    object_name: Optional[str] = None
    target_name: Optional[str] = None
    source_file: Optional[str] = None
    source_demo: Optional[str] = None


class EpisodeSplitter:
    """Splits LIBERO episodes into primitive action segments."""
    
    def __init__(self, min_segment_length: int = 10):
        """Initialize the episode splitter."""
        self.min_segment_length = min_segment_length
        self.primitive_tasks = defaultdict(list)
        
    def detect_gripper_events(self, robot_states: np.ndarray) -> List[Tuple[int, str]]:
        """
        Detect gripper grasp and release events using adaptive thresholds.
        
        Args:
            robot_states: Array of robot states [T, state_dim]
        
        Returns:
            List of (timestep, event_type) tuples
        """
        gripper_pos = robot_states[:, 0]
        
        # Use adaptive thresholds based on data distribution
        gripper_min = np.min(gripper_pos)
        gripper_max = np.max(gripper_pos)
        gripper_range = gripper_max - gripper_min
        
        # Define states: closed (bottom 35%), open (top 35%), transition (middle 30%)
        closed_threshold = gripper_min + 0.35 * gripper_range
        open_threshold = gripper_min + 0.65 * gripper_range
        
        events = []
        prev_state = None
        
        for i in range(len(gripper_pos)):
            if gripper_pos[i] < closed_threshold:
                curr_state = 'closed'
            elif gripper_pos[i] > open_threshold:
                curr_state = 'open'
            else:
                continue  # Skip transition states
            
            # Detect state changes
            if prev_state == 'open' and curr_state == 'closed':
                events.append((i, 'grasp'))
            elif prev_state == 'closed' and curr_state == 'open':
                events.append((i, 'release'))
                
            prev_state = curr_state
        
        return events
    
    def classify_action_after_grasp(self, robot_states: np.ndarray, 
                                   grasp_idx: int, end_idx: int) -> str:
        """
        Classify the action after grasping based on motion patterns.
        
        Args:
            robot_states: Array of robot states
            grasp_idx: Index where grasp occurred
            end_idx: End index of segment
            
        Returns:
            Action type: 'place' or 'putin'
        """
        if grasp_idx >= end_idx - 5:
            return 'place'  # Default if too short
            
        # Analyze vertical motion after grasp
        ee_positions = robot_states[grasp_idx:end_idx, 2:5]  # End-effector position
        
        if len(ee_positions) < 5:
            return 'place'
            
        # Check vertical displacement
        initial_height = ee_positions[0, 2]
        final_height = ee_positions[-1, 2]
        vertical_change = final_height - initial_height
        
        # If significant downward motion, it's likely "put in"
        if vertical_change < -0.05:  # Threshold for downward motion
            return 'putin'
        else:
            return 'place'
    
    def extract_object_names(self, task_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract object and target names from task description."""
        patterns = [
            r'pick up the ([\w\s]+?) and (?:place|put) it (?:in|on) the ([\w\s]+)',
            r'(?:grasp|pick up) the ([\w\s]+)',
        ]
        
        task_lower = task_name.lower()
        
        for pattern in patterns:
            match = re.search(pattern, task_lower)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    obj1 = groups[0].strip().replace(' ', '_')
                    obj2 = groups[1].strip().replace(' ', '_')
                    return obj1, obj2
                elif len(groups) == 1:
                    obj = groups[0].strip().replace(' ', '_')
                    return obj, None
        
        return None, None
    
    def segment_episode(self, robot_states: np.ndarray, task_name: str, 
                       source_file: str, demo_key: str) -> List[ActionSegment]:
        """
        Segment an episode into primitive actions.
        
        Args:
            robot_states: Array of robot states [T, state_dim]
            task_name: Task description
            source_file: Source HDF5 file path
            demo_key: Demo key in source file
            
        Returns:
            List of ActionSegment objects
        """
        segments = []
        
        # Extract object names from task
        primary_obj, target_obj = self.extract_object_names(task_name)
        
        # Detect gripper events
        events = self.detect_gripper_events(robot_states)
        
        if len(events) >= 2:  # Expect grasp followed by release
            # Find first grasp event
            grasp_event = None
            release_event = None
            
            for timestep, event_type in events:
                if event_type == 'grasp' and grasp_event is None:
                    grasp_event = timestep
                elif event_type == 'release' and grasp_event is not None and release_event is None:
                    release_event = timestep
                    break
            
            if grasp_event is not None:
                # Segment 1: Grasp action (start to just after grasp)
                grasp_end = min(grasp_event + 15, len(robot_states))
                segments.append(ActionSegment(
                    action_type='grasp',
                    start_idx=0,
                    end_idx=grasp_end,
                    object_name=primary_obj,
                    target_name=None,
                    source_file=source_file,
                    source_demo=demo_key
                ))
                
                # Segment 2: Place/Putin action (from grasp to end)
                place_start = max(grasp_event - 5, 0)  # Small overlap
                action_type = self.classify_action_after_grasp(
                    robot_states, grasp_event, len(robot_states)
                )
                
                segments.append(ActionSegment(
                    action_type=action_type,
                    start_idx=place_start,
                    end_idx=len(robot_states),
                    object_name=primary_obj,
                    target_name=target_obj,
                    source_file=source_file,
                    source_demo=demo_key
                ))
        
        elif len(events) == 1 and events[0][1] == 'grasp':
            # Only grasp detected, split based on grasp point
            grasp_event = events[0][0]
            
            # Segment 1: Grasp
            grasp_end = min(grasp_event + 15, len(robot_states))
            segments.append(ActionSegment(
                action_type='grasp',
                start_idx=0,
                end_idx=grasp_end,
                object_name=primary_obj,
                target_name=None,
                source_file=source_file,
                source_demo=demo_key
            ))
            
            # Segment 2: Place (assume place if no clear release)
            if grasp_end < len(robot_states) - self.min_segment_length:
                segments.append(ActionSegment(
                    action_type='place',
                    start_idx=max(grasp_event - 5, 0),
                    end_idx=len(robot_states),
                    object_name=primary_obj,
                    target_name=target_obj,
                    source_file=source_file,
                    source_demo=demo_key
                ))
        
        else:
            # No clear events detected - use simple temporal split
            mid_point = len(robot_states) // 2
            
            # First half: Grasp
            segments.append(ActionSegment(
                action_type='grasp',
                start_idx=0,
                end_idx=mid_point + 10,
                object_name=primary_obj,
                target_name=None,
                source_file=source_file,
                source_demo=demo_key
            ))
            
            # Second half: Place
            if mid_point - 10 < len(robot_states) - self.min_segment_length:
                segments.append(ActionSegment(
                    action_type='place',
                    start_idx=mid_point - 10,
                    end_idx=len(robot_states),
                    object_name=primary_obj,
                    target_name=target_obj,
                    source_file=source_file,
                    source_demo=demo_key
                ))
        
        return segments
    
    def create_primitive_task_name(self, action_type: str, object_name: Optional[str],
                                  target_name: Optional[str] = None) -> str:
        """Create task name for primitive action."""
        obj = object_name or 'object'
        
        if action_type == 'grasp':
            return f"pick_up_the_{obj}"
        elif action_type == 'place':
            target = target_name or 'target'
            return f"place_the_{obj}_on_the_{target}"
        elif action_type == 'putin':
            target = target_name or 'target'
            return f"put_the_{obj}_in_the_{target}"
        else:
            return f"{action_type}_the_{obj}"
    
    def generate_instruction(self, segment: ActionSegment) -> str:
        """Generate natural language instruction for segment."""
        def format_name(name):
            return name.replace('_', ' ') if name else 'object'
        
        obj = format_name(segment.object_name)
        
        if segment.action_type == 'grasp':
            return f"pick up the {obj}"
        elif segment.action_type == 'place':
            target = format_name(segment.target_name)
            return f"place the {obj} on the {target}"
        elif segment.action_type == 'putin':
            target = format_name(segment.target_name)
            return f"put the {obj} in the {target}"
        else:
            return f"{segment.action_type} the {obj}"
    
    def process_episode_file(self, input_path: str):
        """Process single HDF5 episode file and collect segments."""
        base_name = os.path.basename(input_path).replace('_demo.hdf5', '')
        task_name = base_name.replace('_', ' ')
        
        with h5py.File(input_path, 'r') as f:
            if 'data' not in f:
                print(f"Warning: No 'data' group in {input_path}")
                return
            
            # Process each demonstration
            demo_count = 0
            for demo_key in f['data'].keys():
                demo_data = f['data'][demo_key]
                robot_states = np.array(demo_data['robot_states'])
                
                # Segment the episode
                segments = self.segment_episode(robot_states, task_name, input_path, demo_key)
                
                # Group segments by primitive task
                for segment in segments:
                    task_key = self.create_primitive_task_name(
                        segment.action_type, segment.object_name, segment.target_name
                    )
                    self.primitive_tasks[task_key].append(segment)
                    
                demo_count += 1
            
            print(f"Processed {demo_count} demos from {base_name}")
    
    def save_primitive_tasks(self, output_dir: str):
        """Save all primitive tasks as separate HDF5 files."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating {len(self.primitive_tasks)} primitive task files:")
        for task_name, segments in sorted(self.primitive_tasks.items()):
            print(f"  {task_name}: {len(segments)} demos")
        
        for task_name, segments in tqdm(self.primitive_tasks.items(), desc="Saving tasks"):
            if not segments:
                continue
                
            output_path = os.path.join(output_dir, f"{task_name}_demo.hdf5")
            
            with h5py.File(output_path, 'w') as out_f:
                data_grp = out_f.create_group('data')
                
                for demo_idx, segment in enumerate(segments):
                    demo_key = f'demo_{demo_idx}'
                    demo_grp = data_grp.create_group(demo_key)
                    
                    # Load source data
                    with h5py.File(segment.source_file, 'r') as src_f:
                        src_demo = src_f['data'][segment.source_demo]
                        
                        # Copy attributes (except language)
                        for attr_name, attr_value in src_demo.attrs.items():
                            if attr_name != 'language_instruction':
                                demo_grp.attrs[attr_name] = attr_value
                        
                        # Set new language instruction
                        instruction = self.generate_instruction(segment)
                        demo_grp.attrs['language_instruction'] = instruction
                        
                        # Copy sliced datasets
                        start, end = segment.start_idx, segment.end_idx
                        
                        for key in src_demo.keys():
                            if key == 'obs':
                                # Handle observation group
                                obs_in = src_demo['obs']
                                obs_out = demo_grp.create_group('obs')
                                for obs_key in obs_in.keys():
                                    obs_data = np.array(obs_in[obs_key])
                                    if len(obs_data) == len(np.array(src_demo['robot_states'])):
                                        # Time-indexed - slice it
                                        obs_out.create_dataset(obs_key, data=obs_data[start:end])
                                    else:
                                        # Non-time data - copy as-is
                                        obs_out.create_dataset(obs_key, data=obs_data)
                            else:
                                # Handle regular datasets
                                dataset = np.array(src_demo[key])
                                if len(dataset) == len(np.array(src_demo['robot_states'])):
                                    # Time-indexed - slice it
                                    demo_grp.create_dataset(key, data=dataset[start:end])
                                else:
                                    # Non-time data - copy as-is
                                    demo_grp.create_dataset(key, data=dataset)


def process_libero_dataset(input_dir: str, output_dir: str, task_suite: Optional[str] = None):
    """Process LIBERO dataset into primitive tasks."""
    
    # Find HDF5 files
    if task_suite:
        pattern = os.path.join(input_dir, task_suite, '*.hdf5')
    else:
        pattern = os.path.join(input_dir, '*_no_noops', '*.hdf5')
    
    hdf5_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # Initialize splitter
    splitter = EpisodeSplitter()
    
    # Process all files
    for hdf5_path in tqdm(hdf5_files, desc="Processing episodes"):
        splitter.process_episode_file(hdf5_path)
    
    # Save results
    if task_suite:
        output_suite_dir = os.path.join(output_dir, task_suite + '_primitives')
    else:
        output_suite_dir = output_dir
    
    splitter.save_primitive_tasks(output_suite_dir)
    print(f"\nProcessing complete! Created files in {output_suite_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split LIBERO episodes into primitive tasks")
    parser.add_argument('--input_dir', type=str, default='/Volumes/Personal/modified_libero_hdf5',
                       help='Input directory with LIBERO HDF5 files')
    parser.add_argument('--output_dir', type=str, default='./libero_primitive_tasks',
                       help='Output directory for primitive task files')
    parser.add_argument('--task_suite', type=str,
                       choices=['libero_goal_no_noops', 'libero_object_no_noops', 
                               'libero_10_no_noops', 'libero_90_no_noops', 'libero_spatial_no_noops'],
                       help='Specific task suite to process')
    
    args = parser.parse_args()
    process_libero_dataset(args.input_dir, args.output_dir, args.task_suite)


if __name__ == '__main__':
    main()