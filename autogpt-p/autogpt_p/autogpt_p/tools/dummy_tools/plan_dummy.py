from autogpt_p.tools.dummy_tools.dummy_planner import DummyPlanner
from autogpt_p.tools.plan import Plan


class PlanDummy(Plan):

    def __init__(self, memory, resulting_plan):
        super().__init__(memory)
        self.planner = DummyPlanner(memory, resulting_plan)

    def execute(self):
        self.memory.planner = self.planner
