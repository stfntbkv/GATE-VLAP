from typing import List

from autogpt_p.state_machine.auto_gpt_p_memory import Memory
from autogpt_p.tools.tool import Tool

from autogpt_p.incremental_goal_memory.incremental_goal_memory import \
    IncrementalGoalMemory

KEYWORD = "CORRECTION"
PARAMETERS = []
DESCRIPTION = ("This tool should be called when the user asks you to remember and do the same next time")



class Correction(Tool):
    '''
    This tool is selected when the human has corrected the generated goal
    Should read the last goal state from the auto_p_memory and then save it in the incremental_goal_memory with the user request

    Memory-Reads: auto_gpt_p_memory
    Memory-Writes: incremental_goal_memory
    '''

    def __init__(self, memory: Memory):
        super().__init__(KEYWORD, PARAMETERS, DESCRIPTION, memory)
        self.new_request = None  # braucht man das hier?
        self.new_user_modified_goal = None
        self.memory = memory

    def get_goal_from_autogpt(self):
        return self.memory.get_current_goal()

    def get_request_from_autogpt(self):
        return self.memory.command_memory[0]

    def abort(self):
        pass

    def _execute(self):
        print("Executing Correction")
        new_user_modified_goal = self.get_goal_from_autogpt()
        new_request = self.get_request_from_autogpt()
        memory = IncrementalGoalMemory('../incremental_goal_memory/known_pairs.csv')
        memory.add_pair(new_request, new_user_modified_goal)
        memory.save_memory()

    def get_executable(self, parameters: List[str]) -> Tool:
        tool = Correction(self.memory)
        tool.executable = True
        tool.parameters = parameters
        return tool


if __name__ == "__main__":
    memory = Memory
    memory.get_current_goal()
