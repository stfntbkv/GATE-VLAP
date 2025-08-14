import pandas as pd

from autogpt_p.incremental_goal_memory.incremental_goal_memory import IncrementalGoalMemory

def save_file_examples_to_goal_memory(filename):
    memory = IncrementalGoalMemory()
    df = pd.read_csv(filename)
    number_of_rows = df.shape[0]
    for i in range(number_of_rows):
        request = df.task[i]
        user_modified_goal = df.desired_goal[i]

        memory.add_pair(request, user_modified_goal)
        memory.save_memory()
memory = IncrementalGoalMemory()
memory.reset_csv()
memory.save_memory()

# memory.add_pair('Help me prepare a salad with cucumber', (And([Predicate("inhand", [Object("milk", Type("milk", [])),
#                                                                         Object("robot", Type("robot", []))])]))))
# save_file_examples_to_goal_memory('evaluation_memory_other_environment.csv')
# save_file_examples_to_goal_memory('evaluation_memory_other_objects.csv')
save_file_examples_to_goal_memory('evaluation_memory_reformated.csv')
# save_file_examples_to_goal_memory('evaluation_memory.csv')