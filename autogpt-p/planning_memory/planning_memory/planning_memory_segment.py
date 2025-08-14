from typing import List


class Parameter:

    def __init__(self, name: str, type: str, skill_parameter: str):
        self.name = name
        self.type = type
        self.skill_parameter = skill_parameter

    def __hash__(self):
        return hash(self.name) + hash(self.type)

    def __str__(self):
        return "{}:{}".format(self.name, self.type)


class Capability:
    def __init__(self, name: str, parameters: List[Parameter], preconditions: str, effects: str, skill: str,
                 provider: str):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effects = effects
        # NOTE: these last two parameters are a reference to our skill system and is only relevant
        # for executing a plan on the robot
        self.skill = skill
        self.provider = provider

    def __eq__(self, other):
        if isinstance(other, Capability):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class RegisteredCapabilities:

    def __init__(self, capabilities: List[Capability]):
        self.capabilities = capabilities


class ActorProfile:

    def __init__(self, name: str, type: str, capabilities: List[Capability]):
        self.name = name
        self.type = type
        self.capabilities = capabilities

    def __eq__(self, other):
        if isinstance(other, ActorProfile):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name + ":" + self.type + str([str(c) for c in self.capabilities])


class Actor:

    def __init__(self, name: str, profile: ActorProfile):
        self.name = name
        self.profile = profile

    def __eq__(self, other):
        if isinstance(other, Actor):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name
