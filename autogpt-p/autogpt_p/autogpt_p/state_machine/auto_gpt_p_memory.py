from __future__ import annotations
from typing import List

from autogpt_p.execution.actor_skill_mapping import ActorSkillMapping
from autogpt_p.helpers.prolog_wrapper import PrologWrapper
from pddl.plan import Plan
from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceMappingDatabase
from object_detection.detection_memory_segment import DetectedObject, ObjectRelation

from autogpt_p.exploration.exploration_memory import ExplorationMemory
from autogpt_p.substitution.substitution_memory import SubstitutionMemory
from autogpt_p.incremental_goal_memory.incremental_goal_memory import IncrementalGoalMemory



class Memory:

    def __init__(self, objects: List[DetectedObject], relations: List[ObjectRelation],
                 exploration_memory: ExplorationMemory, substitution_memory: SubstitutionMemory,
                 oam_db: ObjectAffordanceMappingDatabase, actor_skill_mapping: ActorSkillMapping, planner,
                 command_memory: List[str],
                 dialogue_memory: List[str], incremental_goal_memory: IncrementalGoalMemory):
        self.objects = objects
        self.relations = relations
        self.oam_db = oam_db
        self.actor_skill_mapping = actor_skill_mapping
        self.planner = planner
        self.exploration_memory = exploration_memory
        self.substitution_memory = substitution_memory
        self.incremental_goal_memory = incremental_goal_memory
        self.command_memory = command_memory
        self.dialogue_memory = dialogue_memory
        self.tool_memory = []
        self.last_result = ""

    def get_current_goal(self):
        return self.planner.problem.goal

    def get_current_plan(self):
        return self.planner.recent_plan

    def print_memory_for_gpt(self):
        # [print(str(r)) for r in self.relations]

        return "Objects: " + ",".join([o.class_name + str(o.id) for o in self.objects]) + "\nObject Relations: " + ",".join([r.relation_name + " " +
                                                " ".join([o.class_name + str(o.id) for o in r.related_objects])
                                                for r in self.relations]) + "\n" + str(self.exploration_memory)

    def update_actor_location(self, actor_name: str):
        location = None
        actor = None
        for o in self.objects:
            if o.class_name + str(o.id) == self.exploration_memory.current_location:
                location = o
            if o.class_name + str(o.id) == actor_name:
                actor = o
        if location is None:
            print("WARNING! location " + self.exploration_memory.current_location)
        for r in self.relations:
            if r.relation_name == "at" and r.related_objects[0] == actor:
                r.related_objects[1] = location
                return
        if location and actor:
            self.relations.append(ObjectRelation("at", [actor, location]))

    def update_commands(self):
        self.command_memory = [self.substitution_memory.substitute_in_prompt(c) for c in self.command_memory]

    def update_with_plan(self, plan: Plan):
        prolog = PrologWrapper()
        self.relations = prolog.get_result_from_plan(self.objects, self.relations, plan)

    def reset_memory(self):
        self.objects = []
        self.tool_memory = []
        self.relations = []
        self.command_memory = []
        self.planner = None
        self.substitution_memory.reset()
        self.incremental_goal_memory.reset_csv()

    @classmethod
    def empty_memory(cls) -> Memory:
        return Memory([], [], ExplorationMemory({}, ""), SubstitutionMemory(),
                      ObjectAffordanceMappingDatabase.load_from_data("simulation_classes_.json",
                                                                     "proposed_affordances_alternative.json",
                                                                     "simulation_classes_2.json"),
                      None, None, [], [], IncrementalGoalMemory('../incremental_goal_memory/known_pairs.csv'))
