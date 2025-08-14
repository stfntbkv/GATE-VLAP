from autogpt_p.incremental_goal_memory.incremental_goal_memory import IncrementalGoalMemory

memory = IncrementalGoalMemory()
memory.reset_csv()
memory.save_memory()
