#!/usr/bin/env python3
"""
CLIP-RT Execution Service
FastAPI service for executing actions using CLIP-RT
"""

import os
import sys
import uvicorn
import asyncio
import tempfile
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import traceback
import numpy as np
import time

# Add LIBERO path
sys.path.append("./libero/LIBERO")
sys.path.append("./libero")

from libero.libero import benchmark
from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from clip_rt_utils import get_clip_rt, get_tokenizer
from robot_utils import set_seed_everywhere, get_clip_rt_action

# Suppress TensorFlow warnings
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CLIP-RT Execution Service", version="1.0.0")

# Global variables for model and environment
clip_rt_model = None
preprocess = None
tokenizer = None
current_env = None
current_task_description = None
current_obs = None  # Store current observation

# CLIP-RT prompt template (from official script)
CLIP_RT_PROMPT = "what motion should the robot arm perform to complete the instruction '{}'?"

# Device detection (from official script)
import torch
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class ExecutionRequest(BaseModel):
    instruction: str
    suite: str = "libero_goal"
    max_steps: int = 2000  # Much higher limit to ensure task completion
    task_id: Optional[int] = 0
    episode_id: Optional[int] = 0
    save_video: bool = True
    num_episodes: int = 1  # Number of episodes to run
    verbose_logging: bool = True  # Enable detailed step-by-step logging
    task_context: Optional[str] = None  # Full task context for multi-step plans
    use_task_context: bool = False  # Whether to include task context in prompts
    action_index: Optional[int] = None  # Index for multi-action sequences to prevent video overwriting

class ExecutionResult(BaseModel):
    success: bool
    instruction: str
    steps_taken: int
    execution_time: float
    error_message: Optional[str] = None
    final_state: Optional[Dict] = None

class EnvironmentState(BaseModel):
    suite: str
    task_id: int
    episode_id: int
    is_initialized: bool
    task_description: str

class InitializeRequest(BaseModel):
    suite: str = "libero_goal"
    task_id: int = 0
    episode_id: int = 0
    model_path: str = ""
    chunk_cut: int = 8  # From official script
    unnorm_key: str = ""  # Will be set to suite name

def load_clip_rt_model(model_path: str, task_split: str):
    """Load CLIP-RT model and preprocessing"""
    global clip_rt_model, preprocess, tokenizer
    
    if clip_rt_model is None:
        logger.info(f"Loading CLIP-RT model from {model_path}")
        
        # Validate model file exists
        if not os.path.exists(model_path):
            error_msg = f"Model file not found at path: {model_path}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        try:
            clip_rt_model, preprocess = get_clip_rt(
                model_path=model_path,
                task_split=task_split,
            )
            # Ensure model is in evaluation mode (from official script)
            clip_rt_model.eval()
            # Move to device (should already be done in get_clip_rt, but ensure it)
            clip_rt_model = clip_rt_model.to(DEVICE)
            
            tokenizer = get_tokenizer()
            logger.info(f"CLIP-RT model loaded successfully on device: {DEVICE}")
        except Exception as e:
            logger.error(f"Failed to load CLIP-RT model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return clip_rt_model, preprocess, tokenizer

@app.post("/initialize", response_model=EnvironmentState)
async def initialize_environment(request: InitializeRequest):
    """Initialize LIBERO environment and CLIP-RT model"""
    global current_env, current_task_description, current_obs
    
    logger.info(f"Initializing environment for suite: {request.suite}, task: {request.task_id}")
    
    try:
        # Set random seed for reproducibility (following official script)
        set_seed_everywhere(7)
        
        # Set action un-normalization key (from official script)
        if not request.unnorm_key:
            request.unnorm_key = request.suite
        logger.info(f"Using unnorm_key: {request.unnorm_key}, chunk_cut: {request.chunk_cut}")
        
        # Load CLIP-RT model if not already loaded
        logger.info(f"Model path provided: '{request.model_path}'")
        if request.model_path:
            logger.info(f"Loading CLIP-RT model from: {request.model_path}")
            load_clip_rt_model(request.model_path, request.suite)
        else:
            logger.warning("No model path provided - model will not be loaded")
        
        # Initialize LIBERO task suite
        try:
            benchmark_dict = benchmark.get_benchmark_dict()
            if request.suite not in benchmark_dict:
                available_suites = list(benchmark_dict.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Suite '{request.suite}' not available. Available suites: {available_suites}"
                )
            
            task_suite = benchmark_dict[request.suite]()
            logger.info(f"Initialized task suite: {request.suite} with {task_suite.n_tasks} tasks")
            
            if request.task_id >= task_suite.n_tasks:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Task ID {request.task_id} out of range for suite {request.suite} (max: {task_suite.n_tasks-1})"
                )
            
            # Get task and initial states
            task = task_suite.get_task(request.task_id)
            initial_states = task_suite.get_task_init_states(request.task_id)
            
            logger.info(f"Task {request.task_id}: {task.language}")
            logger.info(f"Available episodes: {len(initial_states)}")
            
            if request.episode_id >= len(initial_states):
                raise HTTPException(
                    status_code=400,
                    detail=f"Episode ID {request.episode_id} out of range (max: {len(initial_states)-1})"
                )
                
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Failed to initialize LIBERO task suite: {e}")
            raise HTTPException(status_code=500, detail=f"LIBERO initialization failed: {str(e)}")
        
        # Initialize environment
        env, task_description = get_libero_env(task, "clip_rt", resolution=256)
        current_env = env
        current_task_description = task_description
        
        # Reset environment BEFORE setting initial state (following official pattern)
        env.reset()
        current_obs = env.set_init_state(initial_states[request.episode_id])
        
        logger.info(f"Environment initialized successfully for task: {task_description}")
        
        return EnvironmentState(
            suite=request.suite,
            task_id=request.task_id,
            episode_id=request.episode_id,
            is_initialized=True,
            task_description=task_description
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute", response_model=ExecutionResult)
async def execute_instruction(request: ExecutionRequest):
    """Execute a single instruction using CLIP-RT"""
    global current_env, current_task_description, clip_rt_model, preprocess, tokenizer, current_obs
    
    logger.info(f"Executing instruction: {request.instruction}")
    
    if current_env is None:
        raise HTTPException(
            status_code=400, 
            detail="Environment not initialized. Call /initialize first."
        )
    
    if clip_rt_model is None:
        raise HTTPException(
            status_code=400,
            detail="CLIP-RT model not loaded. Call /initialize with model_path first."
        )
    
    start_time = time.time()
    steps_taken = 0
    
    try:
        # Use the stored observation from initialization
        obs = current_obs
        
        # Execute the instruction following official LIBERO pattern
        resize_size = 224
        max_steps = request.max_steps
        num_steps_wait = 10  # Official LIBERO wait steps
        replay_images = []  # Store images for video
        t = 0  # Step counter like in official script
        
        while t < max_steps + num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall (following official LIBERO pattern)
                if t < num_steps_wait:
                    obs, _, done, _ = current_env.step(get_libero_dummy_action("clip_rt"))
                    t += 1
                    continue
                # Get preprocessed image (following official pattern)
                img = get_libero_image(obs, resize_size)
                
                # Prepare observation (following official LIBERO pattern)
                # Note: img is already processed by get_libero_image, but we need it as numpy for our observation dict
                observation = {
                    "full_image": img,  # This should be numpy array for get_clip_rt_action
                    "state": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
                }
                
                # Get action from CLIP-RT using official prompt template
                # Use the decomposed instruction directly to test if CLIP-RT can handle it
                effective_instruction = request.instruction
                
                # Optional: Add context as a prefix if requested
                if request.use_task_context and request.task_context:
                    # Provide context but keep the specific instruction
                    # Format: "To [full task], [specific step]"
                    effective_instruction = f"To {request.task_context.lower()}, {request.instruction.lower()}"
                    logger.info(f"Step {t}: Using context-aware instruction: '{effective_instruction}'")
                
                formatted_prompt = CLIP_RT_PROMPT.format(effective_instruction)
                
                if request.verbose_logging or t % 5 == 0:  # Log every 5 steps or if verbose
                    logger.info(f"Step {t}/{max_steps + num_steps_wait}: Using instruction: '{request.instruction}'")
                    if request.task_context:
                        logger.info(f"Step {t}: Task context: '{request.task_context}'")
                    logger.info(f"Step {t}: Original task: '{current_task_description}'")
                    logger.info(f"Step {t}: Formatted prompt: '{formatted_prompt}'")
                
                action_chunks = get_clip_rt_action(
                    clip_rt_model,
                    preprocess,
                    tokenizer,
                    observation,
                    formatted_prompt,  # Use formatted prompt like official script
                )
                
                # Log action chunks like official script
                logger.info(f"Step {t}: Action_chunks: {action_chunks}")
                
                # Validate action chunks format (like official script assertions)
                if not isinstance(action_chunks, list):
                    logger.error(f"Expected list of action chunks, got {type(action_chunks)}")
                elif len(action_chunks) != 8:
                    logger.warning(f"Expected 8 action chunks, got {len(action_chunks)}")
                elif not all(isinstance(chunk, list) and len(chunk) == 7 for chunk in action_chunks):
                    logger.warning(f"Expected chunks of 7 dimensions each")
                else:
                    logger.info(f"✅ Valid action chunks: {len(action_chunks)} chunks of 7 dimensions each")
                
                done_flag = False
                
                # Execute action chunks (following official LIBERO pattern)
                for action_chunk_idx, action_chunk in enumerate(action_chunks):
                    # Get preprocessed image for video
                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)
                    
                    obs, _, done, _ = current_env.step(action_chunk)
                    if done:
                        done_flag = True
                        break
                    t += 1
                
                if done_flag:
                    break
                    
            except Exception as e:
                logger.error(f"Caught exception during execution: {e}")
                import traceback
                traceback.print_exc()
                break
        
        execution_time = time.time() - start_time
        
        # Check if task was completed successfully (following official LIBERO pattern)
        success = done_flag
        
        # Save video if requested
        if request.save_video and replay_images:
            try:
                os.makedirs("./experiments/videos", exist_ok=True)
                # Create unique task description including action index if available
                if request.action_index is not None:
                    # Include action index and instruction in filename to prevent overwriting
                    video_task_desc = f"action_{request.action_index}_{request.instruction}"
                else:
                    video_task_desc = request.instruction
                
                save_rollout_video(
                    request.suite,
                    "1",  # model checkpoint epoch
                    replay_images,
                    1,  # episode number
                    success=success,
                    task_description=video_task_desc,  # Use unique description
                    log_file=None
                )
                logger.info(f"Video saved successfully for action {request.action_index}: {request.instruction}")
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")
        
        logger.info(f"Task execution completed. Success: {success}, Steps: {t}")
        logger.info(f"Task description: {current_task_description}")
        
        return ExecutionResult(
            success=success,
            instruction=request.instruction,
            steps_taken=t,  # Use actual step count like official script
            execution_time=execution_time,
            final_state={
                "done": done_flag,
                "steps": t
            }
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error executing instruction: {e}")
        logger.error(traceback.format_exc())
        
        return ExecutionResult(
            success=False,
            instruction=request.instruction,
            steps_taken=steps_taken,
            execution_time=execution_time,
            error_message=str(e)
        )

@app.post("/execute_sequence")
async def execute_sequence(instructions: List[str], suite: str = "libero_goal", 
                          task_id: int = 0, episode_id: int = 0,
                          task_context: Optional[str] = None,
                          use_task_context: bool = False) -> List[ExecutionResult]:
    """Execute a sequence of instructions"""
    logger.info(f"Executing sequence of {len(instructions)} instructions")
    if task_context:
        logger.info(f"Task context: {task_context}")
    
    results = []
    
    for i, instruction in enumerate(instructions):
        logger.info(f"Executing instruction {i+1}/{len(instructions)}: {instruction}")
        
        request = ExecutionRequest(
            instruction=instruction,
            suite=suite,
            task_id=task_id,
            episode_id=episode_id,
            task_context=task_context,
            use_task_context=use_task_context
        )
        
        result = await execute_instruction(request)
        results.append(result)
        
        # Stop if instruction fails
        if not result.success:
            logger.error(f"Instruction {i+1} failed, stopping sequence")
            break
    
    successful_instructions = sum(1 for r in results if r.success)
    logger.info(f"Sequence completed: {successful_instructions}/{len(instructions)} successful")
    
    return results

@app.get("/environment_state")
async def get_environment_state():
    """Get current environment state"""
    global current_env, current_task_description
    
    if current_env is None:
        return {
            "initialized": False,
            "task_description": None
        }
    
    return {
        "initialized": True,
        "task_description": current_task_description
    }

@app.post("/reset_environment")
async def reset_environment():
    """Reset the current environment"""
    global current_env
    
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    
    try:
        current_env.reset()
        logger.info("Environment reset successfully")
        return {"status": "reset_successful"}
    except Exception as e:
        logger.error(f"Failed to reset environment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "CLIP-RT Execution Service",
        "model_loaded": clip_rt_model is not None,
        "environment_initialized": current_env is not None
    }

@app.get("/suites")
async def list_suites():
    """List available LIBERO suites"""
    return {
        "suites": [
            "libero_goal",
            "libero_10", 
            "libero_90",
            "libero_spatial",
            "libero_object"
        ]
    }

if __name__ == "__main__":
    logger.info("Starting CLIP-RT Execution Service...")
    uvicorn.run(app, host="0.0.0.0", port=8002)