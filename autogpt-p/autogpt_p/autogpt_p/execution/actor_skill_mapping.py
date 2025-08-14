from typing import List, Dict

from planning_memory.planning_memory_segment import Actor, Capability, Parameter

from autogpt_p.execution.actor import from_actor
from pddl.core import Type, Action, Variable, Object
from pddl.problem import parse_logic_element

ACTOR_SUFFIX = "er"
ACTOR_NAME = "actor"


def get_action_type(actor: Type, capability: Capability):
    return Type(_add_actor_suffix(capability.name), [actor])


def make_pddl_action_from_capability(actor_type: Type, object_type: Type, capability: Capability, predicates):
    """
    Turns a skill from the skill memory into a pddl action
    :param object_type:
    :param actor_type:
    :param capability:
    :param predicates:
    :return:
    """
    actor_variable = Variable(ACTOR_NAME, get_action_type(actor_type, capability))
    variables = [actor_variable] + [_parameter_to_variable(p, object_type) for p in capability.parameters]
    return Action(capability.name, variables, parse_logic_element(capability.preconditions, predicates, variables),
                  parse_logic_element(capability.effects, predicates, variables), [0])


class ActorSkillMapping:
    """
    Tries to get an executable skill from memory and returns it
    Responsible for connection from skills to pddl and back
    """

    def __init__(self, actors: List[Actor]):
        self.actors = [from_actor(actor) for actor in actors]

    def get_robot_actors(self):
        return [a for a in self.actors if a.is_robot()]

    def get_human_actors(self):
        return [a for a in self.actors if not a.is_robot()]

    def get_all_capabilities(self):
        capabilities = set()
        for a in self.actors:
            [capabilities.add(c) for c in a.actor.profile.capabilities]
        return list(capabilities)

    def get_skill_with_parameters(self, actor_name: str, capability_name: str, capability_parameters: List[str]):
        assert actor_name in [a.actor.name for a in self.actors]
        actor = [a.actor for a in self.actors if a.actor.name == actor_name][0]
        assert capability_name in [c.name for c in actor.profile.capabilities]
        capability = [c for c in actor.profile.capabilities if c.name == capability_name][0]
        return capability.skill, self.get_skill_parameters(capability, capability_parameters)

    def get_skill_parameters(self, capability, capability_parameters):
        parameters = {}
        for i, param in enumerate(capability.parameters):
            if len(param.skill_parameter) > 0:
                parameters[param.skill_parameter] = capability_parameters[i]
        return parameters

    def get_action_types(self, actor: Type) -> List[Type]:
        return list(self.get_action_type_map(actor).values())

    def get_action_type_map(self, actor: Type) -> Dict[str, Type]:
        capabilities = self.get_all_capabilities()
        return {c.name: get_action_type(actor, c) for c in capabilities}

    def get_profile_types(self, actor_type, robot_type, human_type: Type) -> List[Type]:
        action_types = self.get_action_type_map(actor_type)
        return [Type(actor.actor.profile.name, [action_types[c.name] for c in actor.actor.profile.capabilities]
                     + ([robot_type] if actor.is_robot() else [human_type])) for actor in self.actors]

    def get_actor_objects(self, profile_types: List[Type]):
        profile_types_map = {profile.name: profile for profile in profile_types}
        return [Object(actor.actor.name, profile_types_map[actor.actor.profile.name]) for actor in self.actors]

    def make_pddl_actions(self, actor_type: Type, object_type: Type, predicates):
        return [make_pddl_action_from_capability(actor_type, object_type, c, predicates)
                for c in self.get_all_capabilities()]

    def verbalize_skill(self, actor_name, capability_name, capability_parameters) -> str:
        return capability_name + " " + " ".join(capability_parameters)


def _add_actor_suffix(action_name: str):
    return (action_name + ACTOR_SUFFIX).lower()


def _parameter_to_variable(parameter: Parameter, object_type: Type) -> Variable:
    return Variable(parameter.name, Type(parameter.type, [object_type]))
