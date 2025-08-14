from incremental_goal_learning.helpers.semantic_similarity import get_embeddings
from incremental_goal_learning.incremental_goal_memory.user_modified_goal_mapped_to_user_request import \
    IncrementalGoalMemory
from incremental_goal_learning.helpers.semantic_similarity import cosine_distance
from pddl.problem import *
import pandas as pd

import pandas as pd

x = cosine_distance("Bring a glass of milk", "I want a glass of milk")
print(x)
print("="*100)
y = cosine_distance("I want you to bring the milk into the fridge", "Why is the milk still not in the fridge")
print(y)