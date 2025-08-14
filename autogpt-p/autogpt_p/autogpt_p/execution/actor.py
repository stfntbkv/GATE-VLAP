from planning_memory.planning_memory_segment import Actor, Capability

ROBOT_TYPE = "robot"
HUMAN_TYPE = "human"


class ExecutingActor:

    def __init__(self, actor: Actor):
        self.actor = actor

    def get_name(self):
        return self.actor.name

    def get_capability(self, name) -> Capability:
        assert name in [c.name for c in self.actor.profile.capabilities]
        return [capability for capability in self.actor.profile.capabilities if capability.name == name][0]

    def is_robot(self):
        pass


class Robot(ExecutingActor):

    def __init__(self, actor: Actor):
        super().__init__(actor)

    def is_robot(self):
        return True


class Human(ExecutingActor):

    def __init__(self, actor: Actor):
        super().__init__(actor)

    def is_robot(self):
        return False


def from_actor(actor: Actor) -> ExecutingActor:
    if actor.profile.type == ROBOT_TYPE:
        return Robot(actor)
    else:
        return Human(actor)
