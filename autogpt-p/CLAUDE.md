# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is AutoGPT+P, an autonomous planning agent that combines classical logic planning with GPT-4's creativity and flexibility for robotic task planning. The project uses PDDL (Planning Domain Definition Language) and Fast Downward planner for structured task execution.

## Key Components

### Core Architecture
- **autogpt_p/**: Main package containing the planning agent implementation
- **state_machine/**: State machine implementation for agent behavior control
- **planning/**: PDDL planning logic using Fast Downward and AutoGPTPlanner
- **execution/**: Plan execution and actor skill mapping
- **llm/**: LLM interface abstraction (primarily ChatGPT/OpenAI)
- **tools/**: Various tools including planning, exploration, substitution, and correction
- **evaluation/**: Comprehensive evaluation framework for different scenarios

### Dependencies
The project depends on several local packages:
- **object_detection/**: Object detection functionality
- **object_affordance_mapping/**: Mapping objects to their affordances
- **chat_gpt_oam/**: ChatGPT object affordance mapping
- **planning_memory/**: Planning memory management
- **pddl/**: PDDL parsing and generation
- **downward/**: Fast Downward planner integration

## Environment Setup

Required environment variables:
```bash
export FAST_DOWNWARD_ROOT=$(pwd)/downward
export AUTOGPT_ROOT=$(pwd)/autogpt_p
export OAM_ROOT=$(pwd)/object_affordance_mapping
export OPENAI_API_KEY='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'
export P
```

## Development Commands

### Running Evaluations
```bash
cd autogpt_p/evaluation/
python alternative_suggestion_evaluation.py
python planner_evaluation.py
python autogpt_p_evaluation.py
python incremental_goal_evaluation.py
```

### Testing
Uses pytest for testing:
```bash
pytest
```

## Key Architecture Patterns

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

## Data Organization

### Evaluation Data
- `data/evaluation/scenarios/`: Test scenarios in CSV format
- `data/evaluation/scenes/`: Scene descriptions in text format
- `data/evaluation/allowed_substitutions/`: Object substitution rules
- `logs/`: Evaluation results and logs organized by evaluation type

### Configuration
- Uses Poetry for dependency management
- Configuration objects in evaluation modules control test parameters
- Supports different LLM models (GPT-3, GPT-4, GPT-5) and scenarios

## Important Implementation Details

### PDDL Integration
- Domain and problem files are generated dynamically from scene descriptions
- Generic vs. typed predicates can be toggled for Fast Downward compatibility
- Debug mode saves PDDL files to `pddl_debug_files/` for inspection

### Error Handling
- Validation error handlers provide feedback for goal correction
- Timeout handling for complex planning problems
- Partial planning fallback for unsolvable full plans

### Evaluation Framework
- Multiple evaluation types: planning, alternative suggestions, incremental memory
- CSV-based scenario definitions with configurable parameters
- Comprehensive logging and result tracking