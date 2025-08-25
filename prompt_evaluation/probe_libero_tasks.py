#!/usr/bin/env python3
"""
Probe LIBERO task IDs to see which task each ID loads
"""

import asyncio
import aiohttp
import argparse

async def probe_task(session, execution_url, suite, task_id, model_path):
    """Initialize environment and return task description"""
    init_data = {
        "suite": suite,
        "task_id": task_id,
        "episode_id": 0,
        "model_path": model_path
    }
    
    try:
        async with session.post(f"{execution_url}/initialize", json=init_data) as response:
            if response.status == 200:
                result = await response.json()
                return task_id, result.get('task_description', 'Unknown')
            else:
                return task_id, f"Error: HTTP {response.status}"
    except Exception as e:
        return task_id, f"Error: {str(e)}"

async def probe_all_tasks(execution_url, suite, max_tasks, model_path):
    """Probe all task IDs to see what they load"""
    
    async with aiohttp.ClientSession() as session:
        # Check service health first
        try:
            async with session.get(f"{execution_url}/health") as response:
                if response.status != 200:
                    print("Execution service not healthy")
                    return
        except Exception as e:
            print(f"Cannot connect to execution service: {e}")
            return
        
        print(f"\nProbing {suite} tasks...")
        print("="*60)
        
        # Probe each task ID
        for task_id in range(max_tasks):
            task_id, description = await probe_task(session, execution_url, suite, task_id, model_path)
            print(f"Task ID {task_id}: {description}")
            await asyncio.sleep(0.5)  # Small delay between probes

def main():
    parser = argparse.ArgumentParser(description="Probe LIBERO task IDs")
    parser.add_argument("--suite", default="libero_10", 
                       choices=["libero_10", "libero_goal", "libero_spatial", "libero_object", "libero_90"],
                       help="LIBERO suite to probe")
    parser.add_argument("--max-tasks", type=int, default=10,
                       help="Maximum number of task IDs to probe")
    parser.add_argument("--execution-url", default="http://localhost:8002",
                       help="Execution service URL")
    
    args = parser.parse_args()
    
    # Set model path based on suite
    model_paths = {
        "libero_10": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_10.pt",
        "libero_goal": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_goal.pt",
        "libero_spatial": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_spatial.pt",
        "libero_object": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_object.pt",
        "libero_90": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_90.pt"
    }
    model_path = model_paths.get(args.suite, model_paths["libero_10"])
    
    asyncio.run(probe_all_tasks(args.execution_url, args.suite, args.max_tasks, model_path))

if __name__ == "__main__":
    main()