from __future__ import annotations
import functools
import itertools
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Set


class LogicElement:
    pass


class IndentationManager:

    def __init__(self):
        self.indentation_list = []
        self.current_indentation = 0

    def next_line(self):
        self.indentation_list.append(self.current_indentation)

    def increase_indentation(self):
        self.current_indentation += 1

    def decrease_indentation(self):
        self.current_indentation -= 1

    def format(self, string: str, indentation_length=4):
        lines = string.split("\n")
        # print(len(lines))
        # print(len(self.indentation_list))
        indent = ' ' * indentation_length
        result = ''
        for i in range(len(lines)):
            result += (indent * self.indentation_list[i]) + lines[i] + "\n"
        return result

    def append(self, indentation_manager):
        appended_indentation = [i + self.current_indentation for i in indentation_manager.indentation_list]
        appended = IndentationManager()
        appended.indentation_list.extend(self.indentation_list)
        appended.indentation_list.extend(appended_indentation)
        appended.current_indentation = self.current_indentation + indentation_manager.current_indentation
        return appended


class Type:
    def __init__(self, name: str, supertypes):
        self.name = name
        self.supertypes = supertypes

    def add_supertype(self, supertype):
        if supertype not in self.supertypes:
            self.supertypes.append(supertype)

    def is_subtype(self, supertype):
        return supertype in self.supertypes

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Type):
            return self.name == other.name
        return False

    def __hash__(self):
        return self.name.__hash__()

    def get_all_supertypes(self):
        supertypes = self.supertypes + list(itertools.chain(*[supertype.get_all_supertypes()
                                                              for supertype in self.supertypes]))
        return list(set(supertypes))


class Literal(ABC, LogicElement):

    def __init__(self, name: str, type: Type):
        self.name = name
        self.type = type

    @abstractmethod
    def write_no_type(self):
        pass

    @classmethod
    def from_string(cls, string: str):
        string = string.replace("?", "")
        if string.isdigit():
            return Constant.from_string(string)
        return Object.from_string(string)


class Constant(Literal):
    def __init__(self, value: int):
        super().__init__(str(value), None)
        self.value = value

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False

    def __hash__(self):
        return self.name.__hash__()

    @classmethod
    def from_string(cls, string: str) -> Constant:
        return Constant(int(string))

    def write_no_type(self):
        return self.name


class Variable(Literal):
    def __init__(self, name: str, type: Type):
        super().__init__(name, type)

    def __str__(self):
        return "?" + self.name + " - " + self.type.name

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __hash__(self):
        return self.name.__hash__()

    def write_no_type(self):
        return "?" + self.name


class Object(Literal):

    def __init__(self, name: str, type: Type):
        super().__init__(name, type)

    def __str__(self):
        return self.name + " - " + self.type.name

    def __eq__(self, other):
        if isinstance(other, Object):
            return self.name == other.name
        return False

    def __hash__(self):
        return self.name.__hash__()

    @classmethod
    def from_string(cls, string: str):
        pattern = r'(\D+)(\d+)'

        # Use re.match to find the pattern in the string
        match = re.match(pattern, string)

        if match:
            # Extract non-digits and digits from the matched groups
            class_name = match.group(1)
        else:
            class_name = string
        return Object(string, Type(class_name, []))

    def write_no_type(self):
        return self.name


def print_variable_list(variables: List[Literal], in_definition=False):
    string = ""
    for v in variables:
        string += " " + (str(v) if in_definition else v.write_no_type())
    return string[0:]


class Predicate(LogicElement):

    def __init__(self, name, variables: List[Literal], comment="", definition=False):
        self.name = name
        self.variables = variables
        self.comment = comment
        self.definition = definition

    def in_definition(self, definition=True):
        return Predicate(self.name, self.variables, self.comment, definition)

    def write_no_type(self):
        return str(self.in_definition(False))

    def __str__(self):
        comment = ""
        if self.definition and len(self.comment) > 0:
            comment = "; {}\n".format(self.comment.format(*[v.write_no_type() for v in self.variables]))
        return comment + "(" + self.name + " " + print_variable_list(self.variables, self.definition) + ")"

    def __eq__(self, other):
        if isinstance(other, Predicate):
            return self.name == other.name and self.variables == other.variables
        return False

    def __hash__(self):
        return self.name.__hash__() + sum([v.__hash__() for v in self.variables])

    def __call__(self, new_variables: List[Literal]):
        return Predicate(self.name, new_variables, self.comment, self.definition)

    @classmethod
    def from_string(cls, string: str) -> Predicate:
        elements = string.split(" ")
        predicate_name = elements[0]
        objects = [Object.from_string(s) for s in elements[1:] if s != ""]
        return Predicate(predicate_name, objects)


class Equals(Predicate):

    def __init__(self, left: Literal, right: Literal):
        super().__init__("=", [left, right])

    def __str__(self):
        return "(= " + self.variables[0].write_no_type() + " " + self.variables[1].write_no_type() + ")"


class Function(LogicElement):

    def __init__(self, name: str, arguments: List[LogicElement]):
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return "(" + self.name + " " + " ".join([str(a) for a in self.arguments]) + ")"

    def __eq__(self, other):
        if isinstance(other, Function):
            return self.name == other.name and self.arguments == other.arguments
        return False


class Increase(Function):

    def __init__(self, base_value: LogicElement, increase_by: LogicElement):
        super().__init__("increase", [base_value, increase_by])


class Decrease(Function):

    def __init__(self, base_value: LogicElement, decrease_by: LogicElement):
        super().__init__("decrease", [base_value, decrease_by])


class LogicOp(LogicElement):

    def __init__(self, logic_elements: List[LogicElement]):
        self.logic_elements = logic_elements

    def __str__(self):
        string = ""
        for e in self.logic_elements:
            string += " " + str(e)
        return string[1:]

    @classmethod
    def from_string(self, string: str) -> LogicElement:
        string = remove_redundant_parenthesis(string)
        function_keywords = ("=", "increase", "decrease")
        if string.startswith(('and', "or", "not")):
            i = string.find("(")
            logic_elements = []
            while i >= 0:
                end = find_closing_parenthesis(string, i)
                logic_element_string = string[i + 1:end]
                logic_elements.append(LogicOp.from_string(logic_element_string))
                i = string.find("(", end)
            if string.startswith('and'):
                return And(logic_elements)
            elif string.startswith('or'):
                return Or(logic_elements)
            elif string.startswith('not'):
                return Not(logic_elements[0])
            else:
                return And([])
        elif string.startswith(function_keywords):
            # format can be "keyword () x", "keyword x ()", "keyword x x", keyword () ()
            # clean up the string so only the arguments remain
            arguments_string = string.lstrip()
            keyword = arguments_string.split()[0]
            arguments_string = arguments_string.replace(keyword, "").lstrip()
            i = arguments_string.find("(", 0)
            first = None
            second = None
            while i >= 0:
                end = find_closing_parenthesis(arguments_string, i)
                logic_element_string = arguments_string[i + 1:end]
                elem = LogicOp.from_string(logic_element_string)
                if i == 0:
                    first = elem
                elif end == len(arguments_string) - 1:
                    second = elem
                i = arguments_string.find("(", end)
            args = [a for a in arguments_string.split(" ") if a != ""]
            if first is None:
                first = Literal.from_string(args[0])
            if second is None:
                second = Literal.from_string(args[-1])
            if keyword == "=":
                return Equals(first, second)
            elif keyword == "increase":
                return Increase(first, second)
            elif keyword == "decrease":
                return Decrease(first, second)
        else:
            return Predicate.from_string(string)


class UnaryLogicOp(LogicOp):

    def __init__(self, logic_element: LogicElement):
        super().__init__([logic_element])


class NAryLogicOp(LogicOp):

    def __init__(self, logic_elements: List[LogicElement]):
        super().__init__(logic_elements)


class Not(UnaryLogicOp):
    def __init__(self, logic_element: LogicElement):
        super().__init__(logic_element)

    def __str__(self):
        return "(not " + super().__str__() + ")"


class And(NAryLogicOp):

    def __init__(self, logic_elements: List[LogicElement]):
        super().__init__(logic_elements)

    def __str__(self):
        return "(and " + super().__str__() + ")"


class Or(NAryLogicOp):

    def __init__(self, logic_elements: List[LogicElement]):
        super().__init__(logic_elements)

    def __str__(self):
        return "(or " + super().__str__() + ")"


class QuantifiedOp(LogicElement):

    def __init__(self, variables: List[Literal], element: LogicElement):
        self.variables = variables
        self.element = element

    def __str__(self):
        string = " (" + print_variable_list(self.variables, True)
        return string + ") " + str(self.element)


class ExistsOp(QuantifiedOp):
    def __init__(self, variables: List[Literal], element: LogicElement):
        super().__init__(variables, element)

    def __str__(self):
        return "(exists" + super().__str__() + ")"


class DerivedPredicate:

    def __init__(self, predicate: Predicate, precondition: LogicElement):
        self.predicate = predicate
        self.predicate.definition = False
        self.precondition = precondition

    def __str__(self):
        return "(:derived " + str(self.predicate) + " " + str(self.precondition) + ")"


class Action:

    def __init__(self, name: str, parameters: List[Literal], preconditions: LogicElement, effect: And,
                 actor_indexes=None):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.effect = effect
        self.actor_index = actor_indexes if actor_indexes is not None else [0]

    def __str__(self):
        return "(:action " + self.name + "\n" + ":parameters (" + print_variable_list(self.parameters, True) + ")\n" \
            + ":precondition " + str(self.preconditions) + "\n:effect " + str(self.effect) + "\n)"

    def get_actors(self) -> List[Literal]:
        return [self.parameters[i] for i in self.actor_index]

    def has_actor(self):
        return len(self.actor_index) > 0


class TextAction(Action):
    """
    This class models an action where the effects do not need to be represented as a logicElement but as a string
    If the effects are correctly formatted that makes no difference for the use in pddl
    """

    def __init__(self, name: str, parameters: List[Literal], preconditions: str, effect: str,
                 actor_indexes=None):
        super().__init__(name, parameters, And([]), And([]), actor_indexes)
        self.preconditions_str = preconditions
        self.effect_str = effect

    def __str__(self):
        return "(:action " + self.name + "\n" + ":parameters (" + print_variable_list(self.parameters, True) + ")\n" \
            + ":precondition " + self.preconditions_str + "\n:effect " + self.effect_str + "\n)"


class CostMode(Enum):
    NONE = "NONE"
    MIN_HUMAN = "MIN_HUMAN"
    MIN_ROBOT = "MIN_ROBOT"
    BALANCED = "BALANCED"

    def __str__(self) -> str:
        """Get the string representation."""
        return f":{self.value}"


class RS(Enum):
    """A set of requirements that can be used in PDDL."""

    STRIPS = ":strips"
    TYPING = ":typing"
    NEG_PRECONDITION = ":negative-preconditions"
    DIS_PRECONDITION = ":disjunctive-preconditions"
    UNIVERSAL_PRECONDITION = ":universal-preconditions"
    EXISTENTIAL_PRECONDITION = ":existential-preconditions"
    QUANTIFIED_PRECONDITION = ":quantified-preconditions"
    EQUALITY = ":equality"
    CONDITIONAL_EFFECTS = ":conditional-effects"
    ADL = ":adl"
    DERIVED_PREDICATES = ":derived-predicates"
    NON_DETERMINISTIC = ":non-deterministic"

    def strip(self) -> str:
        """Strip the leading colon."""
        return self.value[1:]


ALL_REQUIREMENTS: Set[str] = {v.value for v in RS}


@functools.total_ordering
class Requirements(Enum):
    """Enum class for the requirements."""

    STRIPS = RS.STRIPS.strip()
    TYPING = RS.TYPING.strip()
    NEG_PRECONDITION = RS.NEG_PRECONDITION.strip()
    DIS_PRECONDITION = RS.DIS_PRECONDITION.strip()
    UNIVERSAL_PRECONDITION = RS.UNIVERSAL_PRECONDITION.strip()
    EXISTENTIAL_PRECONDITION = RS.EXISTENTIAL_PRECONDITION.strip()
    QUANTIFIED_PRECONDITION = RS.QUANTIFIED_PRECONDITION.strip()
    EQUALITY = RS.EQUALITY.strip()
    CONDITIONAL_EFFECTS = RS.CONDITIONAL_EFFECTS.strip()
    ADL = RS.ADL.strip()
    DERIVED_PREDICATES = RS.DERIVED_PREDICATES.strip()
    NON_DETERMINISTIC = RS.NON_DETERMINISTIC.strip()

    @classmethod
    def strips_requirements(cls) -> Set["Requirements"]:
        """Get the STRIPS requirements."""
        return {
            Requirements.TYPING,
            Requirements.NEG_PRECONDITION,
            Requirements.DIS_PRECONDITION,
            Requirements.EQUALITY,
            Requirements.CONDITIONAL_EFFECTS,
        }

    def __str__(self) -> str:
        """Get the string representation."""
        return f":{self.value}"

    def __repr__(self) -> str:
        """Get an unambiguous representation."""
        return f"Requirements{self.name}"

    def __lt__(self, other):
        """Compare with another object."""
        if isinstance(other, Requirements):
            return self.value <= other.value
        else:
            return super().__lt__(other)


def find_closing_parenthesis(string, open_index):
    count = 1
    for i in range(open_index + 1, len(string)):
        if string[i] == "(":
            count += 1
        elif string[i] == ")":
            count -= 1
            if count == 0:
                return i
    return -1


def remove_redundant_parenthesis(string: str) -> str:
    iteration_string = string
    while iteration_string.startswith("("):
        end_index = find_closing_parenthesis(iteration_string, 0)
        iteration_string = iteration_string[1:end_index]
    return iteration_string
