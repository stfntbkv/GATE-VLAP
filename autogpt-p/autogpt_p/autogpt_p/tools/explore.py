from typing import List

from autogpt_p.tools.tool import Tool

KEYWORD = "EXPLORE"
PARAMETERS = ["location"]
DESCRIPTION = "Moves the robot from the current location to the unexplored location {} to look for " \
              "undiscovered objects at that location. (Example Usage: EXPLORE counter3)"

SCENE_PREFIX = "R003/"


class Explore(Tool):
    """
    NOTE TO REVIEWERS: This code contained references to specifically used software that could affect double-blindness


    Moves the robot from the current location to the unexplored location {} to look for " \
    "undiscovered objects at that location.

    Memory-Reads: Exploration-Memory
    Memory-Writes: Exploration-Memory, Object-Memory, Relations-Memory

    Failure-Cases:
        - Location does not exist (internal)
        - Location is already explored (internal)
        - Navigation did not succeed (external)
    """

    def __init__(self, memory):
        super().__init__(KEYWORD, PARAMETERS, DESCRIPTION, memory)

    def get_executable(self, parameters: List[str]) -> Tool:
        tool = Explore(self.memory)
        tool.executable = True
        tool.parameters = parameters
        return tool

    def _execute(self):
        pass

    def _update_objects_correct_ids(self, new_objects):
        id_map = {}
        for o in [o.class_name for o in self.memory.objects]:
            id_map[o] = o.id if (o in id_map.keys() and o.id > id_map[o]) or o not in id_map.keys() else id_map[o]
        for o in new_objects:
            id_map[o.class_name] += 1
            o.id = id_map[o.class_name]
        self.memory.objects += new_objects

    def abort(self):
        return

    def __eq__(self, other):
        return isinstance(other, Explore)
