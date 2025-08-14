from typing import List

from sympy import And as SAnd, Or as SOr, Not as SNot, Symbol, to_dnf
from pddl.core import LogicElement, And, Or, Not, Predicate, Object, Type


def predicate_to_sympy(pred_expression: LogicElement):
    mapping = {}
    return _rec_predicate_to_sympy(pred_expression, mapping), mapping


def _rec_predicate_to_sympy(pred_expression: LogicElement, mapping):
    if isinstance(pred_expression, And):
        return SAnd(*[_rec_predicate_to_sympy(elem, mapping) for elem in pred_expression.logic_elements])
    elif isinstance(pred_expression, Or):
        return SOr(*[_rec_predicate_to_sympy(elem, mapping) for elem in pred_expression.logic_elements])
    elif isinstance(pred_expression, Not):
        return SNot(_rec_predicate_to_sympy(pred_expression.logic_elements[0], mapping))
    elif isinstance(pred_expression, Predicate):
        if pred_expression in mapping.keys():
            return mapping[pred_expression]
        else:
            counter = len(mapping.keys())
            new_symbol = Symbol("P" + str(counter))
            mapping[pred_expression] = new_symbol
            return new_symbol


def sympy_to_predicate(sympy_expression, mapping):
    # if the dnf does not contain any ors there is only one subgoal
    if isinstance(sympy_expression, SAnd):
        result = []
        # the return value needs to be a list of lists in order for the code to work
        for symbol in sympy_expression.args:
            if symbol.is_Not:
                result.append(Not(mapping[symbol.args[0]]))
            else:
                result.append(mapping[symbol])
        return [result]
    elif isinstance(sympy_expression, SNot):
        return [[Not(mapping[sympy_expression.args[0]])]]
    else:
        subgoals = sympy_expression.args
        predicate_subgoals = []
        for subgoal in subgoals:
            # if a subgoal only contains a single predicate this needs to be treated separately
            if subgoal.is_Not:
                predicate_subgoals.append([Not(mapping[subgoal.args[0]])])
            elif subgoal.is_Atom:
                predicate_subgoals.append([mapping[subgoal]])
            else:
                predicates = []
                for symbol in subgoal.args:
                    if symbol.is_Not:
                        predicates.append(Not(mapping[symbol.args[0]]))
                    else:
                        predicates.append(mapping[symbol])
                predicate_subgoals.append(predicates)
        return predicate_subgoals


def goal_to_dnf(goal: LogicElement) -> List[List[Predicate]]:
    # this is a special case where it would not work otherwise
    if isinstance(goal, And) and len(goal.logic_elements) == 1 and isinstance(goal.logic_elements[0], Predicate):
        return[[goal.logic_elements[0]]]
    sympy_expression, mapping = predicate_to_sympy(goal)
    dnf = to_dnf(sympy_expression)
    
    
    mapping = {value: key for key, value in mapping.items()}
    return sympy_to_predicate(dnf, mapping)

if __name__ == "__main__":
    goal = And([Not(Predicate("closed", [Object("door0", Type("door",[]))]))])
    dnf = goal_to_dnf(goal)
    print(dnf)
