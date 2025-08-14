import itertools
from itertools import chain
from typing import Dict, List


from autogpt_p.tools.tool import Tool
from object_detection.detection_memory_segment import DetectedObject, ObjectRelation


from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.state_machine.auto_gpt_p_memory import Memory
from autogpt_p.tools.explore import Explore


class DummyScene:

    def __init__(self, object_locations: Dict[str, List[ObjectRelation]], objects):
        self.object_locations = object_locations
        self.objects = objects

    def explore_with_memory(self, memory: Memory, location: str):
        if location in self.object_locations.keys():
            detected_relations = self.object_locations[location]
            memory.exploration_memory.update_explored(location)
            memory.objects.extend(self.objects[location])
            memory.objects = list(set(memory.objects))
            memory.relations.extend(detected_relations)

    def get_objects_from_relation(self, relations):
        return list(set(chain.from_iterable([[o for o in r.related_objects] for r in relations])))

    def get_objects(self) -> List[DetectedObject]:
        new_objects = []
        for objects in self.objects.values():
            new_objects.extend(objects)
        return new_objects

    def get_relations(self) -> List[ObjectRelation]:
        return list(chain.from_iterable(self.object_locations.values()))

    @classmethod
    def from_simulated_scene(cls, scene: SimulatedScene):
        location_dict = {}
        for l in scene.get_locations():
            location_dict[l] = list(set(itertools.chain.from_iterable([scene.get_all_relations_of(o) for o in scene.get_objects_at(l)])))
        return DummyScene(location_dict, scene.object_locations)

    def __str__(self):
        return "\n".join([k + " , ".join([str(r) for r in v]) for k, v in self.object_locations.items()])


class ExploreDummy(Explore):

    def __init__(self, memory: Memory, dummy_scene: DummyScene, execute=True):
        super().__init__(memory)
        self.dummy_scene = dummy_scene

    def get_executable(self, parameters: List[str]) -> Tool:
        tool = ExploreDummy(self.memory, self.dummy_scene)
        tool.executable = True
        tool.parameters = parameters
        return tool

    def _execute(self):
        if self.parameters[0] in self.memory.exploration_memory.explorable_locations:
            if not self.memory.exploration_memory.is_explored(self.parameters[0]):
                self.dummy_scene.explore_with_memory(self.memory, self.parameters[0])
                # the newly detected objects need to be written to the memory with the correct ids so that the ids are
                # synced up for planning
                return "explored " + self.parameters[0]
            else:
                return self.parameters[0] + " was already explored"
        else:
            return self.parameters[0] + " cannot be explored. Possible destinations: " \
                + ", ".join(self.memory.exploration_memory.get_explored())

