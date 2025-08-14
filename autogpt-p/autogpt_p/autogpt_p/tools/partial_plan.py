from typing import List

from autogpt_p.planning.autogpt_planner import AutoGPTPlanner

from autogpt_p.llm.llm_factory import LLMFactory

from autogpt_p.tools.tool import Tool

KEYWORD = "INCOMPLETE_PLAN"
PARAMETERS = []
DESCRIPTION = "Given the scene memory and the prompt this tool finds a plan that fulfills a sub goal of " \
              "the user request if it cannot be fully achieved with the objects in the scene."


class PartialPlan(Tool):

    def __init__(self, memory):
        super().__init__(KEYWORD, PARAMETERS, DESCRIPTION, memory)

    def get_executable(self, parameters: List[str]) -> Tool:
        tool = PartialPlan(self.memory)
        tool.executable = True
        tool.parameters = parameters
        return tool

    def _execute(self):
        print("Executing Partial Planner")
        llm = LLMFactory.get_instance().produce_llm()
        planner = AutoGPTPlanner(llm, self.memory.objects, self.memory.relations,
                                 list(self.memory.exploration_memory.explorable_locations.keys()), self.memory.oam_db,
                                 self.memory.actor_skill_mapping, partial_plan=True)
        planner.plan_with_incremental_goal_memory(self.memory.command_memory[len(self.memory.command_memory) - 1], 3)
        self.memory.planner = planner
        if planner.recent_plan.is_valid():
            return "Planning succeeded"
        else:
            return "Planning failed due to unreachable goal"

    def abort(self):
        return

    def __eq__(self, other):
        return isinstance(other, PartialPlan)
