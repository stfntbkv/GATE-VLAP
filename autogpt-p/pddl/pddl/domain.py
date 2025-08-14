from typing import List

from pddl.core import IndentationManager, Requirements, Type, Predicate, DerivedPredicate, Action, CostMode, \
    Variable, Increase


class Domain:

    def __init__(self, name, requirements: List[Requirements], types: List[Type], predicates: List[Predicate],
                 derived: List[DerivedPredicate], actions: List[Action], cost_mode=CostMode.NONE,
                 actor_type: Type = None):
        self.name = name
        self.requirements = requirements
        self.types = types
        self.predicates = predicates
        self.derived = derived
        self.actions = actions
        self.cost_mode = cost_mode
        if not self.cost_mode:
            self.functions = []
        else:
            assert actor_type
            self.init_costs(actor_type)
        # self.init_capacity(actor_type)

    def __str__(self):
        indent = IndentationManager()
        header = self.write_header()
        indent.next_line()
        indent.increase_indentation()
        indent.next_line()
        requirements = self.write_requirements()
        indent.next_line()
        indent.next_line()
        types, type_indent = self.write_types()
        indent = indent.append(type_indent)
        indent.next_line()
        predicates, predicates_indent = self.write_predicates()
        indent = indent.append(predicates_indent)
        indent.next_line()
        functions = None
        if self.has_costs():
            functions, functions_indent = self.write_functions()
            indent = indent.append(functions_indent)
            indent.next_line()
        derived, derived_indent = self.write_derived()
        indent = indent.append(derived_indent)
        indent.next_line()
        actions, actions_indent = self.write_actions()
        indent = indent.append(actions_indent)
        indent.indentation_list = indent.indentation_list[:len(indent.indentation_list) - 1]
        indent.decrease_indentation()
        indent.next_line()
        result = header + "\n\n" + requirements + "\n\n" + types + "\n\n" + predicates + \
                 (functions + "\n\n" if self.has_costs() else "") + "\n\n" + derived + "\n\n" + actions + "\n)"
        return indent.format(result)

    def write_header(self):
        return "(define (domain " + self.name + ")"

    def write_requirements(self):
        return "(:requirements " + ' '.join([str(r) for r in self.requirements]) + ")"

    def write_types(self):
        indent = IndentationManager()
        string = "(:types\n"
        indent.next_line()
        indent.increase_indentation()
        subtype_dict = get_all_subtypes(self.types)
        for supertype, subtypes in subtype_dict.items():
            string += ' '.join(subtypes) + " - " + supertype + "\n"
            indent.next_line()
        indent.decrease_indentation()
        indent.next_line()
        return string + ")", indent

    def write_predicates(self):
        string = "(:predicates\n" + '\n'.join(str(p) for p in self.predicates) + "\n)"
        indent = IndentationManager()
        indent.next_line()
        indent.increase_indentation()
        [indent.next_line() for _ in range(len(self.predicates))]
        [indent.next_line() for pred in self.predicates if len(pred.comment) > 0]
        indent.decrease_indentation()
        indent.next_line()
        return string, indent

    def write_derived(self):
        string = "\n\n".join(str(d) for d in self.derived)
        indent = IndentationManager()
        [indent.next_line() for _ in range(len(self.derived) * 2 - 1)]
        return string, indent

    def write_actions(self):
        string = "\n\n".join(str(a) for a in self.actions)
        indent = IndentationManager()
        [self.action_indentation(indent) for _ in range(len(self.actions))]
        return string, indent

    def write_functions(self):
        string = "(:functions\n" + '\n'.join(str(func) for func in self.functions) + "\n)"
        indent = IndentationManager()
        indent.next_line()
        indent.increase_indentation()
        [indent.next_line() for _ in range(len(self.functions))]
        [indent.next_line() for func in self.functions if len(func.comment) > 0]
        indent.decrease_indentation()
        indent.next_line()
        return string, indent

    def action_indentation(self, indent):
        indent.next_line()
        indent.increase_indentation()
        indent.next_line()
        indent.next_line()
        indent.next_line()
        indent.decrease_indentation()
        indent.next_line()
        indent.next_line()

    def has_costs(self):
        return self.cost_mode != CostMode.NONE

    def init_costs(self, actor_type: Type):
        total_cost = Predicate("total-cost", [])
        cost = Predicate("cost", [Variable("actor", actor_type)])
        self.functions = [total_cost, cost.in_definition()]
        for action in self.actions:
            if len(action.actor_index) > 0:
                actor_index = action.actor_index[0]
                action.effect.logic_elements.append(Increase(total_cost, cost([action.parameters[actor_index]])))

    def init_capacity(self, actor_type: Type):
        carry_capacity = Predicate("carry-capacity", [Variable("actor", actor_type)]).in_definition()
        self.functions.append(carry_capacity)


def get_all_subtypes(types: List[Type]):
    subtype_dict = {}
    for type in types:
        for super_type in type.supertypes:
            if super_type.name in subtype_dict.keys():
                subtype_dict[super_type.name].append(type.name)
            else:
                subtype_dict[super_type.name] = [type.name]
    return subtype_dict
