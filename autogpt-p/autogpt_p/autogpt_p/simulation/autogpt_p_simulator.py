from typing import List

import numpy as np
import logging
from autogpt_p.helpers.prolog_wrapper import PrologWrapper

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.execution.plan_executor import PlanExecutorInterface
from autogpt_p.exploration.exploration_memory import ExplorationMemory
from autogpt_p.helpers.scene_read_write import read_scene
from autogpt_p.input.input_manager_factory import InputManagerFactory, COMMAND_LINE
from autogpt_p.llm.llm_factory import LLMFactory
from autogpt_p.planning.autogpt_planner import AutoGPTPlanner
from autogpt_p.state_machine.auto_gpt_p_context import AutoGPTPContext
from autogpt_p.state_machine.auto_gpt_p_memory import Memory
from autogpt_p.tools.correction import Correction
from autogpt_p.tools.dummy_tools.explore_dummy import DummyScene, ExploreDummy
from autogpt_p.tools.partial_plan import PartialPlan
from autogpt_p.tools.plan import Plan
from autogpt_p.tools.suggest_substitution import SuggestSubstitution
from autogpt_p.tools.tool_selector import ToolSelector
from autogpt_p.tools.tool_selector_factory import ToolSelectorFactory

from autogpt_p.execution.actor_skill_mapping import ActorSkillMapping
from planning_memory.dynamic_actor_provider import DynamicActorProvider
from planning_memory.static_capability_provider import StaticCapabilityProvider

class AutoGPTPSimulator(PlanExecutorInterface):

    def __init__(self, memory: Memory, execute=True):

        super().__init__()
        self.llm_factory = LLMFactory.get_instance()

        self.autogptp = AutoGPTPContext(self.llm_factory.produce_llm(), 10, memory,
                                        self)

        self.memory = memory
        self.scene = None
        self.dummy_scene = None
        self.execute_plan = execute

    def reset_memory(self):
        self.memory.reset_memory()
        # just to be sure
        self.autogptp.llm.reset_history()

    def reset_state_machine(self):
        self.autogptp = AutoGPTPContext(self.llm_factory.produce_llm(), 10, self.memory, self)

    def load_scene(self, path):
        objects, relations, locations = read_scene(path)
        self.set_scene(SimulatedScene(objects, relations, locations))

    def set_scene(self, scene: SimulatedScene, start_location=""):
        if start_location == "":
            for relation in scene.relations:
                if relation.relation_name == "at" and relation.related_objects[0].class_name == "robot":
                    start_location = str(relation.related_objects[1])
        # always add human to the known object if one exists in the scene
        for o in scene.objects:
            if o.class_name == "human":
                self.memory.objects.append(o)
        # this needs to be done so that the exploration memory has dummy locations
        location_map = {location: np.zeros(3) for location in scene.object_locations.keys()}
        self.memory.exploration_memory = ExplorationMemory(location_map, start_location)
        self.scene = scene
        self.dummy_scene = DummyScene.from_simulated_scene(self.scene)
        self.memory.planner = AutoGPTPlanner(self.llm_factory.produce_llm(), [], [],
                                             list(scene.object_locations.keys()),
                                             self.memory.oam_db, self.memory.actor_skill_mapping)

        tool_selector = ToolSelector([Plan(self.memory), PartialPlan(self.memory),
                                      ExploreDummy(self.memory, self.dummy_scene, self.execute_plan),
                                      SuggestSubstitution(self.memory), Correction(self.memory)])
        if ToolSelectorFactory.get_instance() is None:
            ToolSelectorFactory(tool_selector)
        else:
            ToolSelectorFactory.get_instance().tool_selector = tool_selector

    def simulate_exploration(self, locations: List[str], reset_exploration=False):
        current_location = self.memory.exploration_memory.current_location
        for l in locations:
            self.memory.exploration_memory.update_explored(l)
            self.dummy_scene.explore_with_memory(self.memory, l)
        self.memory.exploration_memory.current_location = current_location
        # self.writer.commit(self.writer.make_entity_name("Planning"), "Planning", self.memory.objects)

    def simulate_full_exploration(self):
        self.memory.objects = self.scene.objects
        self.memory.relations = self.scene.relations
        self.memory.exploration_memory.explored = \
            [True for i in range(len(self.memory.exploration_memory.explorable_locations))]

    def _execute(self, plan: Plan):
        prolog = PrologWrapper()
        self.scene.relations = prolog.get_result_from_plan(self.scene.objects, self.scene.relations, plan)
        # reset prolog as otherwise the predicates stay in memory forever
        prolog.clear_all()
        # update memory as we assume all steps where executed correctly
        self.memory.relations = prolog.get_result_from_plan(self.scene.objects, self.memory.relations, plan)

    def abort(self):
        return

    def interactive_command(self):
        input_manager = InputManagerFactory.get_instance().produce_input_manager()
        input = ""
        while input != "quit":
            input_manager.wait_for_input()
            input = input_manager.get_input()
            self.autogptp.process_command(input)

    def command(self, command: str):
        print("calling command")
        self.autogptp.process_command(command)


def main():
    # initialize singletons
    InputManagerFactory(COMMAND_LINE)
    LLMFactory("GPT", "4")
    memory = Memory.empty_memory()
    capabilities = StaticCapabilityProvider("../../../planning_memory/data/capabilities/all_capabilities.json")
    capabilities.process_skills()
    actor_provider = DynamicActorProvider("robot0", "robot_profile", "robot", capabilities)
    actor_skill_mapping = ActorSkillMapping([actor_provider.get_actor()])
    memory.actor_skill_mapping = actor_skill_mapping
    simulator = AutoGPTPSimulator(memory)
    # simulator.load_scene("/common/homes/students/uqqbd_mayr/code/h2t/student_projects/ba_marion_mayr/autogpt-p-standalone/autogpt_p/data/evaluation/scenes/simple_kitchen_dizzy.txt")
    # simulator.simulate_exploration(["table0", "counter0", "refrigerator0", "door0", "window0", "sink0"])
    # simulator.load_scene("/common/homes/students/uqqbd_mayr/code/h2t/student_projects/ba_marion_mayr/autogpt-p-standalone/autogpt_p/data/evaluation/scenes/dining_room_prep.txt")
    # simulator.simulate_exploration(["table0", "counter0", "refrigerator0", "basket0"])
    simulator.load_scene("/common/homes/students/uqqbd_mayr/code/h2t/student_projects/ba_marion_mayr/autogpt-p-standalone/autogpt_p/data/evaluation/scenes/keyboard.txt")
    simulator.simulate_exploration(["table0", "counter0", "refrigerator0", "basket0", "desk0"])
    simulator.interactive_command()


if __name__ == "__main__":
    main()
    # logging.basicConfig(filename='correction_tool_tested.log',
    #                     filemode='a',
    #                     level=logging.INFO,
    #                     format='%(message)s')
    #
    # logging.info("This is an info message")
