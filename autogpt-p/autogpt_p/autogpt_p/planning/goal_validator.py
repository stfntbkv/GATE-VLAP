from itertools import chain
from typing import List

from autogpt_p.helpers.dnf import goal_to_dnf
from autogpt_p.helpers.name_helpers import digits_to_letters
from pddl.core import Predicate, Object, Not, Variable, Type

from pyswip import Prolog

GOAL_CORRECTION = "Your previous answer did not contain a goal in the correct format. Please correct your answer"
PREDICATE_CORRECTION = "Your previous answer contained the predicate {}, however this predicate does not " \
                       "exist in the domain. Please correct your answer"
OBJECT_CORRECTION = "Your previous answer contained the object {}, however this object does not exist in the scene. " \
                    "Please correct your answer"
INVALID_PREDICATE_CORRECTION = "The {} argument of the predicate {} needs to have the type {}." \
                               "Please change the object or the predicate. " \
                               "Do not change the type of the object you used."
INVALID_ARITY_CORRECTION = "The predicate {} needs to have {} arguments instead of {}." \
                           "Please change the number of arguments in predicate. " \
                           "Do not change the type of the object you used."
CONTRADICTION = "There is a logical contradiction in the goal. {}. Please correct your answer"
TYPING_CORRECTION = "Your previous answer contained typing with a '?' Which is not allowed in the goal specification. " \
                    "Please correct your answer without the use of '?'"


class PredicateLimitation:

    def __init__(self, nl_description, prolog_query):
        self.nl_description = nl_description
        self.prolog_query = prolog_query


def int_to_ordinal(number):
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return str(number) + suffix


def _format_predicate(predicate: Predicate):
    return "{}({})".format(predicate.name,
                           ",".join([digits_to_letters(l.write_no_type()) for l in predicate.variables]))


def _setup_prolog():
    prolog = Prolog()
    # add all static rules needed for queries later
    prolog.assertz("on_other(X,Y,Z) :- on(X,Z), not(Y=Z)")
    prolog.assertz("in_other(X,Y,Z) :- in(X,Z), not(Y=Z)")
    prolog.assertz("liquid_in_other(X,Y,Z) :- liquid_in(X,Z), not(Y=Z)")
    prolog.assertz("in_other_hand(X,Y,Z) :- inhand(X,Z), not(Y=Z)")
    prolog.assertz("other_object_inhand(X,Y,Z) :- inhand(Y,Z), not(X=Y)")
    prolog.assertz("at_other(X,Y,Z) :- at(X,Z), not(Y=Z)")
    print("After Setup")
    return prolog


class GoalValidator:
    """
    IMPORTANT:
    The Goal Validator assumes that the goal is already syntactically correct so there are no non-existing predicates
    or objects in it. It only validates whether there are Conflicting Goals or Predicates with objects of the wrong type
    Which are sematic errors.
    """

    def __init__(self, predicate_limitations: List[PredicateLimitation], predicates: List[Predicate], goal):
        self.predicate_limitations = predicate_limitations
        self.predicates = predicates
        self.goal = goal
        self.prolog = _setup_prolog()

    def validate(self):
        goal_dnf = goal_to_dnf(self.goal)
        error = self._check_predicates(goal_dnf)
        if error:
            return error
        return self._check_conflicting_goal(goal_dnf)

    def _check_subgoal(self, limitations, subgoal):
        # if one subgoal is achievable the goal itself has no contradiction
        # INFO: this is a question of how to define it exactly
        for limitation in limitations:
            # if one limitation isn't met for a subgoal, the subgoal is invalid
            if not self._check_limitation_for_subgoal(limitation.prolog_query, subgoal):
                return ConflictingGoal(limitation.nl_description)
        return None

    def _check_conflicting_goal(self, goal_dnf):
        error = ConflictingGoal("Dummy")
        for subgoal in goal_dnf:
            error = self._check_subgoal(self.predicate_limitations, subgoal)
            if not error:
                return None
        return error

    def _check_predicates(self, goal_dnf: List[List[Predicate]]):
        all_predicates = list(set([pred.logic_elements[0] if isinstance(pred, Not) else pred
                                   for pred in list(chain.from_iterable(goal_dnf))]))
        for predicate in all_predicates:
            # get same predicate from typed predicates
            # this assumes the goal is syntactically correct so the predicate actually is in predicates
            # print(str(predicate))
            if predicate.name == "=":
                continue
            typed_predicate = list(filter(lambda pred: pred.name == predicate.name, self.predicates))[0]
            if len(typed_predicate.variables) != len(predicate.variables):
                return InvalidArity(predicate, len(predicate.variables))
            for i, (object, type) in enumerate(zip(predicate.variables, typed_predicate.variables)):
                # print(str(type))
                if isinstance(object, Object) and isinstance(type, Variable):
                    if type.type not in object.type.get_all_supertypes():
                        return InvalidPredicate(predicate, type.type, i)
        return None

    def _check_limitation_for_subgoal(self, limitation, subgoal: List[Predicate]):
        self._clear_prolog_predicates()
        for predicate in subgoal:
            if isinstance(predicate, Not):
                # self.prolog.assertz("\+ {}".format(_format_predicate(predicate.logic_elements[0])))
                continue
            # print(_format_predicate(predicate))
            self.prolog.assertz(_format_predicate(predicate))
        return bool(list(self.prolog.query(limitation)))

    def _clear_prolog_predicates(self):
        self.prolog.retractall("on(X,Y)")
        self.prolog.retractall("at(X,Y)")
        self.prolog.retractall("in(X,Y)")
        self.prolog.retractall("liquid_in(X,Y)")
        self.prolog.retractall("inhand(X,Y)")


class ValidationError:

    def __init__(self, error_message):
        self.error_message = error_message

    def print_error_message(self):
        return self.error_message


class UnknownPredicate(ValidationError):
    def __init__(self, predicate_name: str):
        super().__init__(PREDICATE_CORRECTION.format(predicate_name))


class UnknownObject(ValidationError):

    def __init__(self, object_name: str):
        super().__init__(OBJECT_CORRECTION.format(object_name))


class InvalidPredicate(ValidationError):
    def __init__(self, predicate: Predicate, wrong_type: Type, wrong_object_index):
        super().__init__(INVALID_PREDICATE_CORRECTION.format(int_to_ordinal(wrong_object_index + 1), str(predicate),
                                                             wrong_type.name))


class InvalidArity(ValidationError):
    def __init__(self, predicate: Predicate, number_arguments):
        super().__init__(INVALID_ARITY_CORRECTION.format(str(predicate), len(predicate.variables), number_arguments))


class ConflictingGoal(ValidationError):

    def __init__(self, hint: str):
        super().__init__(CONTRADICTION.format(hint))


class MissingGoal(ValidationError):
    def __init__(self):
        super().__init__(GOAL_CORRECTION)


class TypingError(ValidationError):
    def __init__(self):
        super().__init__(TYPING_CORRECTION)
