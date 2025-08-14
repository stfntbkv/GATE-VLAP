import os

AUTOGPT_VAR = "AUTOGPT_ROOT"
AUTOGPT_PYTHON_PACKAGE = os.environ[AUTOGPT_VAR]
DATA_DIR = os.path.join(AUTOGPT_PYTHON_PACKAGE, "data")
EVALUATION_DIR = os.path.join(DATA_DIR, "evaluation")
SCENARIO_DIR = os.path.join(EVALUATION_DIR, "scenarios")
SAYCAN_DIR = os.path.join(EVALUATION_DIR, "saycan")

LOGS_DIR = os.path.join(AUTOGPT_PYTHON_PACKAGE, "logs")
ALTERNATIVE_SUGGESTION_LOGS_DIR = os.path.join(LOGS_DIR, "alternative_suggestion")
PLANNING_LOGS_DIR = os.path.join(LOGS_DIR, "planning")
INCREMENTAL_GOAL_MEMORY_LOGS_DIR = os.path.join(LOGS_DIR, "incremental_goal_memory")
AUTOGPTP_LOGS_DIR = os.path.join(LOGS_DIR, "autogpt_p")

RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
ALTERNATIVE_SUGGESTION_RESULTS_DIR = os.path.join(RESULTS_DIR, "alternative_suggestion")
PLANNING_RESULTS_DIR = os.path.join(RESULTS_DIR, "planning")
INCREMENTAL_GOAL_MEMORY_RESULTS_DIR = os.path.join(RESULTS_DIR, "incremental_goal_memory")
AUTOGPTP_RESULTS_DIR = os.path.join(RESULTS_DIR, "autogpt_p")
