#!/usr/bin/env python3
"""
Simple CLIP-RT test script - test a single instruction on a specific task
"""

import asyncio
import aiohttp
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_instruction(execution_url, instruction, suite, task_id, episode_id, model_path):
    """Test a single instruction with CLIP-RT"""
    
    async with aiohttp.ClientSession() as session:
        # Check service health
        try:
            async with session.get(f"{execution_url}/health") as response:
                if response.status != 200:
                    logger.error("Execution service not healthy")
                    return None
                health = await response.json()
                logger.info(f"Service status: {health}")
        except Exception as e:
            logger.error(f"Cannot connect to execution service: {e}")
            return None
        
        # Initialize environment
        init_data = {
            "suite": suite,
            "task_id": task_id,
            "episode_id": episode_id,
            "model_path": model_path
        }
        
        try:
            async with session.post(f"{execution_url}/initialize", json=init_data) as response:
                if response.status != 200:
                    logger.error(f"Failed to initialize environment: {response.status}")
                    return None
                result = await response.json()
                logger.info(f"Environment initialized with task: {result['task_description']}")
                logger.info(f"Testing instruction: '{instruction}'")
        except Exception as e:
            logger.error(f"Error initializing: {e}")
            return None
        
        # Execute instruction
        exec_data = {
            "instruction": instruction,
            "suite": suite,
            "max_steps": 500,
            "save_video": True,
            "verbose_logging": False
        }
        
        try:
            logger.info(f"Executing: '{instruction}'")
            async with session.post(f"{execution_url}/execute", json=exec_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "instruction": instruction,
                        "environment_task": result.get("task_description", ""),
                        "success": result.get("success", False),
                        "steps_taken": result.get("steps_taken", 0),
                        "video_path": result.get("video_path", None)
                    }
                else:
                    logger.error(f"Execution failed: {response.status}")
                    return {
                        "instruction": instruction,
                        "success": False,
                        "steps_taken": 0,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            logger.error(f"Error executing: {e}")
            return {
                "instruction": instruction, 
                "success": False,
                "steps_taken": 0,
                "error": str(e)
            }

def main():
    parser = argparse.ArgumentParser(description="Simple CLIP-RT instruction tester")
    
    # Required arguments
    parser.add_argument("instruction", type=str, help="The instruction to test")
    
    # Optional arguments
    parser.add_argument("--suite", default="libero_10", 
                       choices=["libero_10", "libero_goal", "libero_spatial", "libero_object", "libero_90"],
                       help="LIBERO suite to use (default: libero_10)")
    parser.add_argument("--task-id", type=int, default=0, 
                       help="Task ID within the suite (default: 0)")
    parser.add_argument("--episode-id", type=int, default=0,
                       help="Episode ID for the task (default: 0)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to CLIP-RT model checkpoint (if not specified, uses default for suite)")
    parser.add_argument("--execution-url", default="http://localhost:8002",
                       help="Execution service URL (default: http://localhost:8002)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results (optional)")
    
    args = parser.parse_args()
    
    # Set default model path based on suite if not provided
    if args.model_path is None:
        model_paths = {
            "libero_10": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_10.pt",
            "libero_goal": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_goal.pt",
            "libero_spatial": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_spatial.pt",
            "libero_object": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/epoch_3-2.pt",
            "libero_90": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_90.pt"
        }
        args.model_path = model_paths.get(args.suite, model_paths["libero_10"])
    
    # Run the test
    result = asyncio.run(test_instruction(
        args.execution_url,
        args.instruction,
        args.suite,
        args.task_id,
        args.episode_id,
        args.model_path
    ))
    
    if result:
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Instruction: {result['instruction']}")
        if 'environment_task' in result:
            print(f"Environment Task: {result['environment_task']}")
        print(f"Success: {'✅' if result['success'] else '❌'}")
        print(f"Steps Taken: {result['steps_taken']}")
        if result.get('video_path'):
            print(f"Video saved to: {result['video_path']}")
        if result.get('error'):
            print(f"Error: {result['error']}")
        print("="*60)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        print("Test failed to run")
        return 1
    
    return 0 if result and result['success'] else 1

if __name__ == "__main__":
    exit(main())