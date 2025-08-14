from typing import List

from planning_memory.actor_provider import ActorProvider
from planning_memory.capability_provider import CapabilityProvider
from planning_memory.planning_memory_segment import Capability

class DynamicActorProvider(ActorProvider):
    """
    Provides an actor with the capabilities that the given provider offers.
    To get all skills that are currently available, the capability provider should be dynamic
    """

    def __init__(self, name, profile_name, type, capability_provider: CapabilityProvider):
        self.capability_provider = capability_provider
        super().__init__(name, profile_name, type, [capability_provider])

    def _get_capabilities(self) -> List[Capability]:
        return self.capability_provider.get_registered_capability().capabilities
