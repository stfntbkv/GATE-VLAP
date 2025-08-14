from autogpt_p.planning.autogpt_planner import AutoGPTPlanner


class DummyPlanner(AutoGPTPlanner):

    def __init__(self, memory, resulting_plan):
        super().__init__(None, memory.objects, memory.relations, memory.exploration_memory.explorable_locations,
                         memory.oam_db)
        self.resulting_plan = resulting_plan

    def update_scene(self, objects, relations, locations=None):
        return

    def reset_history(self):
        return

    def plan(self, user_task):
        return self.resulting_plan

