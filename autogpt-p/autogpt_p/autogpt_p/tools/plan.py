from typing import List

from autogpt_p.llm.llm_factory import LLMFactory
from autogpt_p.planning.autogpt_planner import AutoGPTPlanner
from autogpt_p.tools.tool import Tool

KEYWORD = "PLAN"
PARAMETERS = []
DESCRIPTION = "Given the scene memory and the prompt this tool finds the plan the fulfills the user request. You can " \
              "only use objects for the plan that are in the List Objects: do not use unexplored locations in the plan"


class Plan(Tool):
    """
    Given the scene memory and the prompt this tool finds the plan the fulfills the user request.

    Memory-Reads: Command-Memory, Object-Memory, Relations-Memory, Exploration-Memory
    Memory-Writes: Planner-Memory

    Failure-Cases:
        - Max-Iterations reached for planning (internal)
    """

    def __init__(self, memory):
        super().__init__(KEYWORD, PARAMETERS, DESCRIPTION, memory)

    def get_executable(self, parameters: List[str]) -> Tool:
        tool = Plan(self.memory)
        tool.executable = True
        tool.parameters = parameters
        return tool

    def _execute(self):
        print("Executing Planner")
        llm = LLMFactory.get_instance().produce_llm()
        planner = AutoGPTPlanner(llm, self.memory.objects, self.memory.relations,
                                 list(self.memory.exploration_memory.explorable_locations.keys()), self.memory.oam_db,
                                 self.memory.actor_skill_mapping)
        planner.plan_with_incremental_goal_memory(self.memory.command_memory[len(self.memory.command_memory) - 1],3)
        self.memory.planner = planner
        if planner.recent_plan.is_valid():
            return "Planning succeeded"
        else:
            return "Planning failed"

    def abort(self):
        return

    def __eq__(self, other):
        return isinstance(other, Plan)
