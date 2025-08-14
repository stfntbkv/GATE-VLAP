import string
from itertools import chain
from typing import List
import os

from autogpt_p.helpers.dnf import goal_to_dnf
from autogpt_p.helpers.name_helpers import digits_to_letters
from pddl.core import Action, LogicElement, And, Not
from object_detection.detection_memory_segment import DetectedObject

from object_detection.detection_memory_segment import ObjectRelation
from pyswip import Prolog

from pddl.core import Predicate
from pddl.plan import Plan

EXECUTION = "execute"

current_module_path = os.path.abspath(__file__)
current_module_dir = os.path.dirname(current_module_path)
package_root_dir = os.path.dirname(current_module_dir)
root_dir = os.path.dirname(package_root_dir)
data_dir = os.path.join(root_dir, 'data/prolog_environment')
environment_file_name = 'execution.txt'


class PrologPredicate:

    def __init__(self, name, number_parameters, parameters=None):
        self.name = name
        self.number_parameters = number_parameters
        if not parameters:
            self.parameters = []
        else:
            self.parameters = parameters

    def print_for_clearing(self):
        return "{}({})".format(self.name, ",".join(get_variables(self.number_parameters)))

    def __str__(self):
        if len(self.parameters) == 0:
            return self.print_for_clearing()
        return "{}({})".format(self.name, ",".join(self.parameters))

    def to_pddl(self) -> Predicate:
        pass

    def to_object_relation(self) -> ObjectRelation:
        pass

    @classmethod
    def from_pddl(cls, predicate: Predicate):
        return PrologPredicate(predicate.name, len(predicate.variables), [digits_to_letters(v.name)
                                                                          for v in predicate.variables])

    @classmethod
    def from_object_relation(cls, relation: ObjectRelation):
        return PrologPredicate(relation.relation_name, len(relation.related_objects),
                               [str(digits_to_letters(o.class_name)) + str(o.id) for o in relation.related_objects])

    @classmethod
    def from_string(cls, string: str):
        string = string.replace(" ", "")
        start = string.find('(')
        name = string[:start]
        parameters = string[start+1:-1].split(",")
        return PrologPredicate(name, len(parameters), parameters)




def _element_str(pred, add):
    selection = "add" if add else "del"
    return "{}({})".format(selection, str(pred))


class PrologAddDelList:

    def __init__(self):
        self.predicates = []
        self.add_or_del = []

    def append_add(self, pred: PrologPredicate):
        self.append(pred, True)

    def append_del(self, pred: PrologPredicate):
        self.append(pred, False)

    def append(self, pred: PrologPredicate, add=True):
        self.predicates.append(pred)
        self.add_or_del.append(add)

    def __str__(self):
        return "[{}]".format(",".join(_element_str(pred, add) for pred, add in zip(self.predicates, self.add_or_del)))


def get_variables(n: int) -> List[str]:
    alphabet = string.ascii_uppercase
    return list(alphabet[:n])


def extract_definition(statement: str) -> str:
    end_index = statement.find(":-")
    return statement[:end_index].replace(" ", "")


def predicate_from_string(definition: str) -> PrologPredicate:
    name = definition[:definition.find("(")]
    n_args = len(definition.split(","))
    return PrologPredicate(name, n_args)


class PrologWrapper:

    def __init__(self):
        self.added_predicates = []
        self.prolog = Prolog()

    def clear_all(self):
        for pred in self.added_predicates:
            self.prolog.retractall(pred.print_for_clearing())

    def add_statement_from_string(self, statement: str):
        self.prolog.assertz(statement)
        self.added_predicates.append(predicate_from_string(extract_definition(statement)))

    def query_bool_from_string(self, query: str):
        return bool(list(self.prolog.query(query)))

    def check_goal_state(self, relations: List[ObjectRelation], goal):
        self.clear_all()
        for r in relations:
            self.add_object_relation(r)
        dnf = goal_to_dnf(goal)
        for subgoal in dnf:
            if self.query_bool_from_string(",".join([self._predicate_to_prolog_string(pred) for pred in subgoal])):
                return True
        return False

    def get_result_from_plan(self, object_list: List[DetectedObject], start_state: List[ObjectRelation], plan: Plan) -> List[ObjectRelation]:
        # setup the environment
        self.clear_all()
        self._setup_plan_environment()
        # prepare finding the objects again when remapping the objects from prolog output
        all_objects = list(chain.from_iterable([[o for o in relation.related_objects] for relation in start_state]))
        all_objects = list(set(all_objects + object_list))
        object_map = {digits_to_letters(o.class_name) + str(o.id): o for o in all_objects}

        relations = [PrologPredicate.from_object_relation(rel) for rel in start_state]
        # [print(action) for action in plan.actions]
        plan_effects = [self._effect_to_prolog_list(action.get_action_effects()) for action in plan.actions]
        # [print(action_effects) for action_effects in plan_effects]
        plan_effects_string = self._list_to_prolog([str(action_effects) for action_effects in plan_effects])
        query = "{}({},{},X)".format(EXECUTION, self._list_to_prolog(relations), plan_effects_string)
        # print(query)
        query_result = list(self.prolog.query(query))[0]
        # print(result)
        prolog_result_state = [PrologPredicate.from_string(r) for r in query_result['X']]
        result_state = [self._object_relation_from_prolog(predicate, object_map) for predicate in prolog_result_state]
        return result_state

    def validate_plan(self, object_list: List[DetectedObject], start_state: List[ObjectRelation], plan: Plan, goal) -> bool:
        return self.check_goal_state(self.get_result_from_plan(object_list, start_state, plan), goal)

    def add_object_relation(self, relation: ObjectRelation):
        self.prolog.assertz(
            "{}({})".format(relation.relation_name, ",".join([o.class_name for o in relation.related_objects])))
        self.added_predicates.append(PrologPredicate(relation.relation_name, len(relation.related_objects)))

    def add_pddl_predicate(self, predicate: Predicate):
        self.prolog.assertz(
            "{}({})".format(predicate.name, ",".join([digits_to_letters(o.write_no_type()) for o in predicate.variables])))
        self.added_predicates.append(PrologPredicate(predicate.name, len(predicate.variables)))

    def _setup_plan_environment(self):
        with open(os.path.join(data_dir, environment_file_name), 'r') as file:
            for line in file:
                self.add_statement_from_string(line.strip())

    def _predicate_to_prolog_string(self, predicate: Predicate) -> str:
        pass

    def _object_relation_from_prolog(self, predicate: PrologPredicate, object_map):
        return ObjectRelation(predicate.name, [object_map[o] for o in predicate.parameters])

    def _list_to_prolog(self, list):
        return "[{}]".format(','.join([str(elem) for elem in list]))

    def _effect_to_prolog_list(self, effect: LogicElement):
        add_del_list = PrologAddDelList()
        if isinstance(effect, And):
            for pred in effect.logic_elements:
                if isinstance(pred, Predicate):
                    add_del_list.append_add(PrologPredicate(pred.name, len(pred.variables),
                                                            [digits_to_letters(v.name) for v in pred.variables]))
                elif isinstance(pred, Not):
                    pred = pred.logic_elements[0]
                    if isinstance(pred, Predicate):
                        add_del_list.append_del(PrologPredicate(pred.name, len(pred.variables),
                                                                [digits_to_letters(v.name) for v in pred.variables]))
        return add_del_list

