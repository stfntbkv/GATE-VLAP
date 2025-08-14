import os
from typing import List, Tuple

from planning_memory.capability_provider import CapabilityProvider, PLANNING_MEMORY_PACKAGE, CAPABILITIES_DIR

CAPABILITIES_FILE = os.path.join(PLANNING_MEMORY_PACKAGE, CAPABILITIES_DIR, "all_capabilities.json")


class StaticCapabilityProvider(CapabilityProvider):
    """
    Ths class takes a list of predefined capabilities that aren't dynamically mapped to an existing skill
    The skills these capabilities can be mapped to can be a generic "do nothing skill" or a skill that
    lets the robot tell the human what they need to do next
    """
    """
    This class should read all skills of the current robot and write a capability for it into the memory
    """

    def __init__(self, capabilities_file_path: str):
        print("Start static")
        super().__init__(capabilities_file_path)


    def _get_skill_index(self, skill_name) -> int:
        return 0

    def _get_skills(self) -> List[Tuple[str, str]]:
        return [("DefaultSkill", "default_skill_provider")]
