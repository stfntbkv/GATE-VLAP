from abc import ABC, abstractmethod
from typing import List

from planning_memory.capability_provider import CapabilityProvider
from planning_memory.planning_memory_segment import Capability, Actor, ActorProfile


class ActorProvider(ABC):

    def __init__(self, name, profile_name, type, capability_providers: List[CapabilityProvider]):
        super().__init__()
        self.name = name
        self.profile_name = profile_name
        self.type = type
        self.capability_providers = capability_providers

    def process_capabilities(self):
        pass

    def get_actor(self):
        return Actor(self.name, ActorProfile(self.profile_name, self.type, self._get_capabilities()))


    @abstractmethod
    def _get_capabilities(self) -> List[Capability]:
        pass
