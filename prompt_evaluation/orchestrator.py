#!/usr/bin/env python3
"""
GATE-VLAP Orchestrator
Coordinates between AutoGPT+P Planning Service and CLIP-RT Execution Service
"""

import asyncio
import aiohttp
import logging
import argparse
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrchestratorConfig:
    planning_service_url: str = "http://localhost:8001"
    execution_service_url: str = "http://localhost:8002"
    clip_rt_model_path: str = "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_10.pt"  # Default model path
    timeout: int = 300  # 5 minutes default timeout

class GateVlapOrchestrator:
    """Orchestrator for the GATE-VLAP pipeline"""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_services_health(self) -> Dict[str, bool]:
        """Check if both services are healthy"""
        health_status = {
            "planning_service": False,
            "execution_service": False
        }
        
        try:
            # Check planning service
            async with self.session.get(f"{self.config.planning_service_url}/health") as response:
                if response.status == 200:
                    health_status["planning_service"] = True
                    logger.info("Planning service is healthy")
                else:
                    logger.error(f"Planning service unhealthy: {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to planning service: {e}")
        
        try:
            # Check execution service
            async with self.session.get(f"{self.config.execution_service_url}/health") as response:
                if response.status == 200:
                    health_status["execution_service"] = True
                    logger.info("Execution service is healthy")
                else:
                    logger.error(f"Execution service unhealthy: {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to execution service: {e}")
        
        return health_status
    
    async def initialize_execution_environment(self, suite: str, task_id: int = 0, episode_id: int = 0) -> bool:
        """Initialize the CLIP-RT execution environment"""
        logger.info(f"Initializing execution environment for {suite}, task {task_id}, episode {episode_id}")
        
        try:
            init_data = {
                "suite": suite,
                "task_id": task_id,
                "episode_id": episode_id,
                "model_path": self.config.clip_rt_model_path
            }
            
            logger.info(f"Sending model path: '{self.config.clip_rt_model_path}'")
            
            async with self.session.post(
                f"{self.config.execution_service_url}/initialize",
                json=init_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Environment initialized: {result['task_description']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to initialize environment: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error initializing environment: {e}")
            return False
    
    async def generate_plan(self, task: str, suite: str = "libero_goal", model: str = "GPT_4") -> Optional[List[Dict]]:
        """Generate a plan using AutoGPT+P"""
        logger.info(f"Generating plan for task: {task}")
        
        try:
            plan_data = {
                "task": task,
                "suite": suite,
                "model": model
            }
            
            async with self.session.post(
                f"{self.config.planning_service_url}/plan",
                json=plan_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result["success"]:
                        logger.info(f"Generated plan with {len(result['actions'])} actions")
                        return result["actions"]
                    else:
                        logger.error(f"Planning failed: {result.get('error_message', 'Unknown error')}")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Planning request failed: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating plan: {e}")
            return None
    
    async def execute_action(self, instruction: str, suite: str = "libero_goal", max_steps: int = 2000,
                           task_context: Optional[str] = None, use_task_context: bool = True,
                           action_index: Optional[int] = None) -> Dict[str, Any]:
        """Execute a single action using CLIP-RT"""
        logger.info(f"Executing action: {instruction}")
        if task_context:
            logger.info(f"With task context: {task_context}")
        
        try:
            execution_data = {
                "instruction": instruction,
                "suite": suite,
                "max_steps": max_steps,
                "task_context": task_context,
                "use_task_context": use_task_context,
                "action_index": action_index
            }
            
            async with self.session.post(
                f"{self.config.execution_service_url}/execute",
                json=execution_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Action executed - Success: {result['success']}, Steps: {result['steps_taken']}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Execution request failed: {error_text}")
                    return {
                        "success": False,
                        "instruction": instruction,
                        "steps_taken": 0,
                        "execution_time": 0,
                        "error_message": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return {
                "success": False,
                "instruction": instruction,
                "steps_taken": 0,
                "execution_time": 0,
                "error_message": str(e)
            }
    
    async def execute_full_pipeline(self, task: str, suite: str = "libero_goal", 
                                  model: str = "GPT_4", task_id: int = 0, episode_id: int = 0,
                                  use_context: bool = True) -> Dict[str, Any]:
        """Execute the complete GATE-VLAP pipeline"""
        logger.info(f"Starting full pipeline for task: {task}")
        
        start_time = time.time()
        
        # Check service health
        health = await self.check_services_health()
        if not all(health.values()):
            return {
                "success": False,
                "error": "One or more services are not healthy",
                "health_status": health
            }
        
        # Initialize execution environment
        if not await self.initialize_execution_environment(suite, task_id, episode_id):
            return {
                "success": False,
                "error": "Failed to initialize execution environment"
            }
        
        # Generate plan
        plan = await self.generate_plan(task, suite, model)
        if not plan:
            return {
                "success": False,
                "error": "Failed to generate plan"
            }
        
        # Execute plan step by step with task context
        execution_results = []
        successful_actions = 0
        
        # Use the original task as context for all sub-actions
        task_context = task
        
        for i, action in enumerate(plan):
            logger.info(f"Executing action {i+1}/{len(plan)}: {action['natural_language']}")
            
            # Pass task context and action index to help CLIP-RT understand the overall goal
            result = await self.execute_action(
                action['natural_language'], 
                suite,
                task_context=task_context if use_context else None,
                use_task_context=use_context,
                action_index=i+1  # Use 1-based indexing for clarity
            )
            execution_results.append(result)
            
            if result['success']:
                successful_actions += 1
                logger.info(f"Action {i+1} completed successfully")
            else:
                logger.error(f"Action {i+1} failed: {result.get('error_message', 'Unknown error')}")
                # Continue with remaining actions (don't break)
                # In some cases, later actions might still succeed
        
        total_time = time.time() - start_time
        success_rate = successful_actions / len(plan) if plan else 0
        
        # Overall success: all actions must succeed for complete task success
        overall_success = successful_actions == len(plan)
        
        result = {
            "success": overall_success,
            "task": task,
            "suite": suite,
            "model": model,
            "plan": plan,
            "execution_results": execution_results,
            "metrics": {
                "total_actions": len(plan),
                "successful_actions": successful_actions,
                "success_rate": success_rate,
                "total_execution_time": total_time
            }
        }
        
        logger.info(f"Pipeline completed - Overall success: {overall_success}")
        logger.info(f"Success rate: {successful_actions}/{len(plan)} ({success_rate:.1%})")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        return result
    
    async def run_libero_evaluation(self, suite: str = "libero_goal", model: str = "GPT_4", 
                                   num_tasks: int = 5, num_episodes_per_task: int = 1) -> Dict[str, Any]:
        """Run evaluation on multiple LIBERO tasks"""
        logger.info(f"Running LIBERO evaluation on {suite} with {num_tasks} tasks")
        
        # Get available tasks - based on actual LIBERO_GOAL tasks from tasks_info.txt
        # Task IDs correspond to the order in tasks_info.txt (0-indexed)
        sample_tasks = {
            "libero_goal": [
                "put the wine bottle on top of the cabinet",      # task_id 0
                "open the top drawer and put the bowl inside",   # task_id 1
                "turn on the stove",                            # task_id 2
                "put the bowl on top of the cabinet",           # task_id 3
                "put the bowl on the plate",                    # task_id 4
                "put the wine bottle on the rack",             # task_id 5
                "put the cream cheese in the bowl",            # task_id 6
                "open the middle drawer of the cabinet",       # task_id 7
                "push the plate to the front of the stove",    # task_id 8
                "put the bowl on the stove"                    # task_id 9
            ],
            "libero_10": [
                "put the apple in the basket",
                "stack the plates",
                "open the drawer",
                "put the mug on the tray",
                "close the dishwasher"
            ]
        }
        
        tasks = sample_tasks.get(suite, sample_tasks["libero_goal"])[:num_tasks]
        
        evaluation_results = []
        
        for task_id, task in enumerate(tasks):
            logger.info(f"Evaluating task {task_id+1}/{len(tasks)}: {task}")
            
            task_results = []
            
            for episode_id in range(num_episodes_per_task):
                logger.info(f"Episode {episode_id+1}/{num_episodes_per_task}")
                
                result = await self.execute_full_pipeline(
                    task=task,
                    suite=suite,
                    model=model,
                    task_id=task_id,
                    episode_id=episode_id
                )
                
                task_results.append(result)
            
            evaluation_results.append({
                "task_id": task_id,
                "task": task,
                "episodes": task_results
            })
        
        # Calculate aggregate metrics
        total_episodes = len(tasks) * num_episodes_per_task
        successful_episodes = sum(
            1 for task_result in evaluation_results
            for episode in task_result["episodes"]
            if episode["success"]
        )
        
        overall_success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
        
        logger.info(f"Evaluation completed - Overall success rate: {successful_episodes}/{total_episodes} ({overall_success_rate:.1%})")
        
        return {
            "suite": suite,
            "model": model,
            "num_tasks": len(tasks),
            "num_episodes_per_task": num_episodes_per_task,
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "overall_success_rate": overall_success_rate,
            "task_results": evaluation_results
        }

async def main():
    parser = argparse.ArgumentParser(description="GATE-VLAP Orchestrator")
    parser.add_argument("--task", type=str, help="Single task to execute")
    parser.add_argument("--suite", type=str, default="libero_goal", help="LIBERO suite")
    parser.add_argument("--model", type=str, default="GPT_4", help="Planning model")
    parser.add_argument("--task-id", type=int, default=0, help="LIBERO task ID for environment")
    parser.add_argument("--evaluation", action="store_true", help="Run full evaluation")
    parser.add_argument("--num-tasks", type=int, default=5, help="Number of tasks for evaluation")
    parser.add_argument("--clip-rt-model", type=str, default="", help="Path to CLIP-RT model")
    parser.add_argument("--planning-url", type=str, default="http://localhost:8001", help="Planning service URL")
    parser.add_argument("--execution-url", type=str, default="http://localhost:8002", help="Execution service URL")
    parser.add_argument("--no-context", action="store_true", help="Run without task context (pure decomposition)")
    
    args = parser.parse_args()
    
    # Use provided model path or default
    model_path = args.clip_rt_model if args.clip_rt_model else "/Users/stefantabakov/Desktop/GATE-VLAP/clip-rt/clip-rt/cliprt_libero_goal.pt"
    
    config = OrchestratorConfig(
        planning_service_url=args.planning_url,
        execution_service_url=args.execution_url,
        clip_rt_model_path=model_path
    )
    
    async with GateVlapOrchestrator(config) as orchestrator:
        if args.evaluation:
            # Run full evaluation
            result = await orchestrator.run_libero_evaluation(
                suite=args.suite,
                model=args.model,
                num_tasks=args.num_tasks
            )
            
            # Save results
            output_file = f"evaluation_results_{args.suite}_{args.model}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Evaluation completed - Results saved to {output_file}")
            print(f"Overall success rate: {result['overall_success_rate']:.1%}")
            
        elif args.task:
            # Run single task
            use_context = not args.no_context
            logger.info(f"Running with context: {use_context}")
            
            result = await orchestrator.execute_full_pipeline(
                task=args.task,
                suite=args.suite,
                model=args.model,
                task_id=args.task_id,
                use_context=use_context
            )
            
            print(f"Task: {args.task}")
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"Actions executed: {result['metrics']['successful_actions']}/{result['metrics']['total_actions']}")
                print(f"Execution time: {result['metrics']['total_execution_time']:.2f}s")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        else:
            # Just check service health
            health = await orchestrator.check_services_health()
            print("Service Health:")
            for service, status in health.items():
                print(f"  {service}: {'✅ Healthy' if status else '❌ Unhealthy'}")

if __name__ == "__main__":
    asyncio.run(main())