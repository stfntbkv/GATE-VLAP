from typing import List

from planning_memory.actor_provider import ActorProvider
from planning_memory.planning_memory_segment import Capability


class StaticActorProvider(ActorProvider):
    """
    Provides an actor with the capabilities that are in the list.
    To get the mapping of the capabilities to the memory ID, this class
    searches through all Provider Segments in the Capability Memory and takes the best match
    The provider prioritizes dynamic capabilities
    """

    def __init__(self, name, profile_name, type, capability_list, cabability_providers, dynamic_provider):
        super().__init__(name, profile_name, type, cabability_providers)
        self.capability_list = capability_list
        self.dynamic_provider= dynamic_provider

    def _get_capabilities(self) -> List[Capability]:
        dynamic_provider = None
        if self.dynamic_provider in self.capability_providers:
            dynamic_provider = self.dynamic_provider
        # append the dynamic provider as the last element, so it overrides all capability mappings
        if dynamic_provider:
            self.capability_providers.append(dynamic_provider)

        # this makes sure that no name duplicates are there
        found_capabilities = {}
        for provider in self.capability_providers:
            capabilities = provider.get_registered_capability().capabilities
            for c in capabilities:
                if c.name in self.capability_list:
                    found_capabilities[c.name] = c

        return list(found_capabilities.values())
