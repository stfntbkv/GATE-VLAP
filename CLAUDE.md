# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GATE-VLAP combines two major research projects for vision-language-action planning in robotics:

1. **AutoGPT+P**: An autonomous planning agent that combines classical PDDL planning with LLM flexibility for robotic task planning
2. **CLIP-RT**: A vision-language-action model for generalist manipulation policies using contrastive imitation learning

## Repository Structure

```
GATE-VLAP/
├── autogpt-p/              # AutoGPT+P planning agent
│   ├── autogpt_p/          # Main package
│   ├── LIBERO/             # LIBERO benchmark integration
│   ├── downward/           # Fast Downward planner
│   └── test_*.py           # LIBERO conversion tests
└── clip-rt/                # CLIP-RT vision-language model
    └── clip-rt/
        ├── libero/         # LIBERO evaluation scripts
        ├── open_clip/      # OpenCLIP implementation
        └── pretrain/       # Pretraining utilities
```

## Development Commands

### AutoGPT+P Setup and Testing

```bash
# Build Fast Downward planner
cd autogpt-p/downward/
python build.py

# Run LIBERO evaluations
cd autogpt-p/
./run_libero_evaluation.sh GPT_5 --suite LIBERO_GOAL

# Test LIBERO conversions
python test_libero_goal_conversion.py
python test_libero_10_conversion.py
python test_libero_90_conversion.py
```

### CLIP-RT Evaluation

```bash
cd clip-rt/clip-rt/libero/
python run_libero_eval_clip_rt.py \
    --model_family clip_rt \
    --pretrained_checkpoint <path> \
    --task_suite_name libero_goal
```

## Environment Variables

```bash
export FAST_DOWNWARD_ROOT=$(pwd)/autogpt-p/downward
export AUTOGPT_ROOT=$(pwd)/autogpt-p/autogpt_p
export OAM_ROOT=$(pwd)/autogpt-p/object_affordance_mapping
export PYTHON_3_8_16=$(which python)
export OPENAI_API_KEY='<your-key>'
export GOOGLE_API_KEY='<your-key>'
```


## Key Architecture Components

### AutoGPT+P
- **State Machine**: Controls agent behavior transitions
- **PDDL Planning**: Generates and executes classical plans via Fast Downward
- **LLM Integration**: Supports GPT-3/4/5 and Gemini models
- **Tool System**: Modular tools for planning, exploration, substitution
- **Memory Management**: Tracks objects, relations, and exploration state
- **Evaluation Framework**: Comprehensive testing on LIBERO and custom scenarios


### CLIP-RT
- **Contrastive Learning**: Trains on vision-language-action triplets
- **Motion Primitives**: Maps language to robot actions
- **Inference Pipeline**: Encodes images and instructions to predict actions
- **LIBERO Integration**: Evaluates on LIBERO benchmark tasks


## Key Architecture Patterns for AutoGPT+P

### State Machine Pattern
The agent uses a state machine (`AutoGPTPContext`) that transitions between states based on user commands and planning results. States handle command processing and determine next transitions.

### Tool-Based Architecture
Tools are modular components that can be selected and executed:
- `Plan`: Full planning tool
- `PartialPlan`: Partial planning for complex tasks
- `ExploreDummy`: Environment exploration
- `SuggestSubstitution`: Object substitution suggestions
- `Correction`: Error correction and refinement

### Memory Management
- `Memory`: Central memory management for objects, relations, and exploration state
- `ExplorationMemory`: Tracks explored locations and current position
- `IncrementalGoalMemory`: Manages incremental learning of planning goals

### Planning Flow
1. User provides natural language task
2. LLM converts task to PDDL goal specification
3. Validation against predicate limitations
4. Fast Downward generates execution plan
5. Plan executor simulates or executes actions


## LIBERO Benchmark Support

Both projects integrate with LIBERO benchmark:
- **Task Suites**: LIBERO_GOAL, LIBERO_10, LIBERO_90, LIBERO_SPATIAL, LIBERO_OBJECT
- **BDDL Conversion**: Converts LIBERO BDDL files to AutoGPT+P format
- **Evaluation Scripts**: Automated evaluation on LIBERO tasks

## Dependencies

Key Python packages:
- `openai`: GPT model integration
- `google-generativeai`: Gemini model support
- `torch`, `transformers`: Neural network models
- `open_clip`: CLIP implementation
- `sympy`: Symbolic computation for planning
- `pandas`: Data handling
- `matplotlib`: Visualization

## Testing and Debugging

- **PDDL Debug Files**: Saved to `pddl_debug_files/` when debug mode enabled
- **Evaluation Logs**: Stored in `autogpt-p/logs/` and `libero_logs/`
- **Results**: CSV outputs in `libero_results/` and evaluation directories
- **Test Scripts**: `test_*_conversion.py` files validate LIBERO integration

## Important Notes

- Always check API keys are set before running evaluations
- LIBERO evaluations require significant compute resources
- Fast Downward must be built before running AutoGPT+P
- Results append to existing logs - copy originals before re-running
- Use appropriate LLM model based on task complexity and budget