import json
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from planning_memory.planning_memory_segment import Capability, Parameter, RegisteredCapabilities

AUTOGPT_VAR = "AUTOGPT_ROOT"
AUTOGPT_PACKAGE = os.environ[AUTOGPT_VAR]
PLANNING_MEMORY_PACKAGE = os.path.join(AUTOGPT_PACKAGE, "python", "planning_memory")
CAPABILITIES_DIR = "data/capabilities"

CAPABILITIES_FILE = os.path.join(PLANNING_MEMORY_PACKAGE, CAPABILITIES_DIR, "all_capabilities.json")


class CapabilityProvider(ABC):
    """
    This class should read all skills of the current robot and write a capability for it into the memory
    """

    def __init__(self, capabilities_file_path: str):
        super().__init__()
        self.registered_capabilities = None
        self.skills = None
        with open(capabilities_file_path, "r") as f:
            json_data = json.load(f)

        # Convert JSON data back to list of tuples
        self.capabilities = []
        for item in json_data:
            name = item["name"]
            parameters = [{"name": param["name"], "type": param["type"], "skill_parameter": param["skill_parameter"]}
                          for param in item["parameters"]]
            preconditions = item["preconditions"]
            effects = item["effects"]
            skill = item["skill"]
            self.capabilities.append((name, parameters, preconditions, effects, skill))

    def process_skills(self):
        print("processing skill")
        self.skills = self._get_skills()
        print(len(self.skills))
        capabilities_memory = []
        skill_names = [s[0] for s in self.skills]

        for c in self.capabilities:
            if c[4] in skill_names:
                idx = self._get_skill_index(c[4])
                capabilities_memory.append(
                    Capability(c[0], [Parameter(p["name"], p["type"], p["skill_parameter"]) for p in c[1]],
                               c[2], c[3], c[4], self.skills[idx][1]))

        self.registered_capabilities = RegisteredCapabilities(capabilities_memory)


    def get_registered_capability(self) -> RegisteredCapabilities:
        return self.registered_capabilities


    @abstractmethod
    def _get_skill_index(self, skill_name) -> int:
        pass

    @abstractmethod
    def _get_skills(self) -> List[Tuple[str, str]]:
        pass
