#!/usr/bin/env python3
"""
Test different instruction formats with CLIP-RT to see what it understands
"""

import asyncio
import aiohttp
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test different instruction formats for LIBERO_10 tasks (similar but different from original suite)
TEST_INSTRUCTIONS = {
    "libero_10_ketchup_milk": {  # Similar to basket tasks - put both the ketchup and the milk in the basket
        "original": "put both the alphabet soup and the cream cheese box in the basket",
        
        # Decomposed subtasks - basic
        "basic_grasp_ketchup": "Pick up the ketchup",
        "basic_grasp_milk": "Pick up the milk",
        "basic_put_both": "Put both items in the basket",
        
        # Goal-oriented subtasks
        "goal_grasp_ketchup": "Grasp the ketchup to place it in the basket",
        "goal_grasp_milk": "Grasp the milk to place it in the basket",
        "goal_place_ketchup": "Place the ketchup in the basket",
        "goal_place_milk": "Place the milk in the basket",
        
        # Context-enriched subtasks
        "context_grasp_ketchup": "To put both items in the basket, first pick up the ketchup",
        "context_grasp_milk": "After placing the ketchup, pick up the milk",
        
        # Motion primitive style
        "primitive_grasp_ketchup": "Execute grasp motion on ketchup with target destination basket",
        "primitive_grasp_milk": "Execute grasp motion on milk with target destination basket",
        
        # Spatial instructions
        "spatial_grasp_ketchup": "Grasp ketchup at current position for basket placement",
        "spatial_grasp_milk": "Grasp milk at current position for basket placement",
        
        # Step-by-step with context
        "step1": "Step 1 of 3: Pick up the ketchup",
        "step2": "Step 2 of 3: Put the ketchup in the basket",
        "step3": "Step 3 of 3: Pick up the milk and put it in the basket",
        
        # Variations
        "ketchup_only": "Pick up the ketchup",
        "milk_only": "Pick up the milk",
        "collect_items": "Collect both the ketchup and milk for the basket"
    },
    
    "libero_10_bowl_drawer": {  # Similar to placement tasks but with drawer - put the black bowl in the bottom drawer of the cabinet and close it
        "original": "put the black bowl in the bottom drawer of the cabinet and close it",
        
        # Decomposed subtasks - basic
        "basic_grasp_bowl": "Pick up the black bowl",
        "basic_open_drawer": "Open the bottom drawer",
        "basic_put_drawer": "Put the bowl in the drawer",
        "basic_close_drawer": "Close the drawer",
        
        # Goal-oriented subtasks
        "goal_grasp_bowl": "Grasp the black bowl to place it in the drawer",
        "goal_open_drawer": "Open the bottom drawer to place the bowl",
        "goal_place_bowl": "Place the black bowl in the bottom drawer",
        "goal_close_drawer": "Close the drawer after placing the bowl",
        
        # Context-enriched subtasks
        "context_open": "To store the bowl, first open the bottom drawer",
        "context_place": "With the drawer open, place the black bowl inside",
        
        # Step-by-step
        "step1": "Step 1 of 4: Open the bottom drawer of the cabinet",
        "step2": "Step 2 of 4: Pick up the black bowl",
        "step3": "Step 3 of 4: Put the bowl in the drawer", 
        "step4": "Step 4 of 4: Close the drawer"
    },
    
    "libero_10_dual_moka": {  # Similar to stove task but with two objects - put both moka pots on the stove
        "original": "put both moka pots on the stove",
        
        # Decomposed subtasks - basic
        "basic_grasp_first": "Pick up the first moka pot",
        "basic_grasp_second": "Pick up the second moka pot",
        "basic_put_both": "Put both moka pots on the stove",
        
        # Goal-oriented subtasks
        "goal_grasp_first": "Grasp the first moka pot to place it on the stove",
        "goal_grasp_second": "Grasp the second moka pot to place it on the stove",
        "goal_place_first": "Place the first moka pot on the stove",
        "goal_place_second": "Place the second moka pot on the stove",
        
        # Context-enriched subtasks
        "context_first": "To place both moka pots, first pick up one moka pot",
        "context_second": "After placing the first moka pot, place the second one",
        
        # Motion primitive style
        "primitive_grasp_first": "Execute grasp motion on first moka pot with target destination stove",
        "primitive_grasp_second": "Execute grasp motion on second moka pot with target destination stove",
        
        # Spatial instructions
        "spatial_grasp_first": "Grasp first moka pot at current position for stove placement",
        "spatial_grasp_second": "Grasp second moka pot at current position for stove placement",
        
        # Step-by-step with context
        "step1": "Step 1 of 3: Pick up the first moka pot",
        "step2": "Step 2 of 3: Place it on the stove",
        "step3": "Step 3 of 3: Pick up the second moka pot and place it on the stove",
        
        # Variations
        "first_only": "Pick up the first moka pot",
        "second_only": "Pick up the second moka pot",
        "prepare_stove": "Prepare the stove for both moka pots"
    }
}

async def test_instruction(session, execution_url, instruction, task_id=0, episode_id=0):
    """Test a single instruction with CLIP-RT"""
    
    # Initialize environment
    init_data = {
        "suite": "libero_10",
        "task_id": task_id,
        "episode_id": episode_id,
        "model_path": "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_10.pt"
    }
    
    try:
        async with session.post(f"{execution_url}/initialize", json=init_data) as response:
            if response.status != 200:
                logger.error(f"Failed to initialize environment: {response.status}")
                return None
            result = await response.json()
            logger.info(f"Initialized: {result['task_description']}")
            logger.info(f"Environment has task: {result['task_description']}")
            logger.info(f"Testing with instruction: '{instruction}'")
    except Exception as e:
        logger.error(f"Error initializing: {e}")
        return None
    
    # Execute instruction
    exec_data = {
        "instruction": instruction,
        "suite": "libero_10",
        "max_steps": 300,  # LIBERO_10 needs more steps
        "save_video": True,
        "verbose_logging": False
    }
    
    try:
        logger.info(f"Testing: '{instruction}'")
        async with session.post(f"{execution_url}/execute", json=exec_data) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "instruction": instruction,
                    "success": result.get("success", False),
                    "steps": result.get("steps_taken", 0)
                }
            else:
                logger.error(f"Execution failed: {response.status}")
                return {
                    "instruction": instruction,
                    "success": False,
                    "steps": 0
                }
    except Exception as e:
        logger.error(f"Error executing: {e}")
        return {
            "instruction": instruction,
            "success": False,
            "steps": 0,
            "error": str(e)
        }

async def run_tests(execution_url="http://localhost:8002", test_set="libero_10_ketchup_milk"):
    """Run all test instructions and compare results"""
    
    results = {}
    test_instructions = TEST_INSTRUCTIONS[test_set]
    
    async with aiohttp.ClientSession() as session:
        # Check service health
        try:
            async with session.get(f"{execution_url}/health") as response:
                if response.status != 200:
                    logger.error("Execution service not healthy")
                    return
                health = await response.json()
                logger.info(f"Service status: {health}")
        except Exception as e:
            logger.error(f"Cannot connect to execution service: {e}")
            return
        
        # Test each instruction format
        for name, instruction in test_instructions.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {name}")
            logger.info(f"Instruction: '{instruction}'")
            logger.info(f"{'='*60}")
            
            # Set task_id based on test_set - map to actual LIBERO_10 task indices  
            if "ketchup_milk" in test_set:
                task_id = 1  # LIVING_ROOM_SCENE1 - put both the alphabet soup and the cream cheese box in the basket
            elif "bowl_drawer" in test_set:
                task_id = 4  # KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
            elif "dual_moka" in test_set:
                task_id = 8  # KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
            else:
                task_id = 0
            result = await test_instruction(session, execution_url, instruction, task_id=task_id)
            if result:
                results[name] = result
                logger.info(f"Result: Success={result['success']}, Steps={result['steps']}")
            
            # Small delay between tests
            await asyncio.sleep(2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    # Group by instruction type
    groups = {
        "Original": ["original"],
        "Basic Decomposed": ["basic_grasp_ketchup", "basic_grasp_milk", "basic_put_both", "basic_grasp_bowl", "basic_open_drawer", "basic_put_drawer", "basic_close_drawer", "basic_grasp_first", "basic_grasp_second"],
        "Goal-Oriented": ["goal_grasp_ketchup", "goal_grasp_milk", "goal_place_ketchup", "goal_place_milk", "goal_grasp_bowl", "goal_open_drawer", "goal_place_bowl", "goal_close_drawer", "goal_grasp_first", "goal_grasp_second", "goal_place_first", "goal_place_second"],
        "Context-Enriched": ["context_grasp_ketchup", "context_grasp_milk", "context_open", "context_place", "context_first", "context_second"],
        "Motion Primitives": ["primitive_grasp_ketchup", "primitive_grasp_milk", "primitive_grasp_first", "primitive_grasp_second"],
        "Spatial": ["spatial_grasp_ketchup", "spatial_grasp_milk", "spatial_grasp_first", "spatial_grasp_second"],
        "Step-by-Step": ["step1", "step2", "step3", "step4"],
        "Variations": ["ketchup_only", "milk_only", "collect_items", "first_only", "second_only", "prepare_stove"]
    }
    
    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if key in results:
                r = results[key]
                status = "✅" if r["success"] else "❌"
                print(f"  {status} {key:20s}: {r['instruction'][:50]}...")
                if r["success"]:
                    print(f"      -> Succeeded in {r['steps']} steps")
    
    # Calculate success rates
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    print(f"\nOverall Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
    
    # Save results to file
    with open("clip_rt_instruction_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to clip_rt_instruction_test_results.json")

def main():
    parser = argparse.ArgumentParser(description="Test CLIP-RT with different instruction formats")
    parser.add_argument("--execution-url", default="http://localhost:8002", help="Execution service URL")
    parser.add_argument("--test-set", default="libero_10_ketchup_milk", help="Which test set to run (libero_10_ketchup_milk, libero_10_bowl_drawer, or libero_10_dual_moka)")
    
    args = parser.parse_args()
    
    asyncio.run(run_tests(args.execution_url, args.test_set))

if __name__ == "__main__":
    main()