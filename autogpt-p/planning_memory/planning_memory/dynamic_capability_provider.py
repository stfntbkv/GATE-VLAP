import os
from typing import List, Tuple

from planning_memory.capability_provider import CapabilityProvider, CAPABILITIES_DIR, PLANNING_MEMORY_PACKAGE

CAPABILITIES_FILE = os.path.join(PLANNING_MEMORY_PACKAGE, CAPABILITIES_DIR, "all_capabilities.json")


class DynamicCapabilityProvider(CapabilityProvider):
    """
    This class should read all skills of the current robot and write a capability for it into the memory
    NOTE TO REVIEWERS: This class only makes sense as the interface to a robotic system that dynamically represents is skills
    in a memory system. As we do not want to show that memory system to be conform with double-blindness this class
    os left as a stub
    """

    def __init__(self, capabilities_file_path: str):
        super().__init__(capabilities_file_path)
        self._skill_manager = None

    def _get_skill_index(self, skill_name) -> int:
        pass

    def _get_skills(self) -> List[Tuple[str, str]]:
        pass
