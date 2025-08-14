from typing import List
from pddl.core import Action, Object, LogicElement, And, Predicate, Not


class ExecutedAction:

    def __init__(self, action: Action, objects: List[Object], cost=1.0):
        self.action = action
        self.objects = objects
        self.cost = cost

    def __str__(self):
        return self.action.name.lower() + " " + ' '.join(o.name for o in self.objects)

    @classmethod
    def from_string(cls, string: str, available_actions: List[Action], available_objects: List[Object]):
        string = string.replace("(", "").replace(")", "")
        tokens = string.split(" ")
        action_name = tokens[0]
        object_names = tokens[1:]

        action = next((action for action in available_actions if action.name.lower() == action_name), None)
        objects = [next((object for object in available_objects if object.name == name), None) for name in object_names]

        return ExecutedAction(action, objects)

    def get_actors(self) -> List[Object]:
        return [self.objects[i] for i in self.action.actor_index]

    def get_action_effects(self) -> And:
        return And([self._replace_variable_with_object(elem) for elem in self.action.effect.logic_elements])

    def _replace_variable_with_object(self, elem: LogicElement) -> LogicElement:
        if isinstance(elem, Not):
            elem = elem.logic_elements[0]
            if isinstance(elem, Predicate):
                return Not(self._replace_variable_in_predicate_with_object(elem))
        elif isinstance(elem, Predicate):
            return self._replace_variable_in_predicate_with_object(elem)

    def _replace_variable_in_predicate_with_object(self, pred: Predicate) -> Predicate:
        return Predicate(pred.name, [self.objects[self.action.parameters.index(v)] for v in pred.variables])


class Plan:

    def __init__(self, actions: List[ExecutedAction]):
        self.costs = float(sum(a.cost for a in actions))
        self.actions = actions

    def is_plan(self):
        return len(self.actions) > 0

    def is_valid(self):
        return self.costs >= 0

    def get_real_length(self):
        sum = 0
        for a in self.actions:
            if a.action.has_actor():
                sum += 1
        return sum

    def __str__(self):
        return '\n'.join(a.__str__() for a in self.actions) + '\n; cost = ' + str(self.costs) + " (union cost)"

    @classmethod
    def from_string(cls, string: str, available_actions: List[Action], available_objects: List[Object]):
        lines = string.split('\n')
        actions = [ExecutedAction.from_string(line, available_actions, available_objects)
                   for line in lines[:(len(lines) - 2)]]
        return Plan(actions)
