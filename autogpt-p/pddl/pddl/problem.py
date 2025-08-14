from typing import List

from pddl.core import Object, Predicate, LogicElement, IndentationManager, Type, And, Or, Not, Literal, \
    Equals, CostMode, Constant, Increase, Decrease, find_closing_parenthesis
from pddl.domain import Domain

PROBLEM = "problem"
DOMAIN = ":domain"
OBJECTS = ":objects"
INIT = ":init"
GOAL = ":goal"


class GoalParsingException(Exception):
    pass


class PredicateParsingException(GoalParsingException):

    def __init__(self, predicate_name, predicates):
        self.predicate_name = predicate_name
        self.predicates = predicates


class ObjectParsingException(GoalParsingException):

    def __init__(self, object_name, objects):
        self.object_name = object_name
        self.objects = objects


class Problem:
    def __init__(self, name, domain: Domain, objects: List[Object], init: List[Predicate], goal: LogicElement,
                 robot_type: Type = None, human_type: Type = None):
        self.name = name
        self.domain = domain
        self.objects = objects
        self.init = init
        self.goal = goal
        self.minimize_cost = domain.has_costs()
        self.init_costs(robot_type, human_type)
        # self.init_capacity(robot_type, human_type)

    def __str__(self):
        return self.write()

    def write_header(self):
        return "(define (problem " + self.name + ")"

    def write_domain(self):
        return "(:domain " + self.domain.name + ")"

    def write_metric(self):
        return "(:metric minimize (total-cost))"

    def write_objects(self):
        string = "(:objects\n" + '\n'.join(str(o) for o in self.objects) + "\n)"
        indent = IndentationManager()
        indent.next_line()
        indent.increase_indentation()
        [indent.next_line() for _ in range(max(1, len(self.objects)))]
        indent.decrease_indentation()
        indent.next_line()
        return string, indent

    def write_init(self):
        string = "(:init \n" + '\n'.join(str(p) for p in self.init) + "\n)"
        indent = IndentationManager()
        indent.next_line()
        indent.increase_indentation()
        [indent.next_line() for _ in range(max(1, len(self.init)))]
        indent.decrease_indentation()
        indent.next_line()
        return string, indent

    def write(self, include_goal=True):
        indent = IndentationManager()
        header = self.write_header()
        indent.next_line()
        indent.next_line()
        indent.increase_indentation()
        domain = self.write_domain()
        indent.next_line()
        indent.next_line()
        metric = None
        objects, objects_indent = self.write_objects()
        indent = indent.append(objects_indent)
        indent.next_line()
        init, init_indent = self.write_init()
        indent = indent.append(init_indent)
        indent.next_line()
        if include_goal:
            goal = self.write_goal() + "\n"
            indent.next_line()
        else:
            goal = ""
        if self.minimize_cost:
            metric = self.write_metric()
            indent.next_line()
            indent.next_line()
        indent.decrease_indentation()
        indent.next_line()
        result = header + "\n\n" + domain + "\n\n" + \
                 objects + "\n\n" + init + "\n\n" + goal + (metric + "\n\n" if metric else "") + ")"
        return indent.format(result)

    def write_goal(self):
        return "(:goal " + self.goal.__str__() + ")"

    @classmethod
    def from_string(cls, string: str, domain: Domain):

        s = string.find(PROBLEM)
        problem_string = string[s:find_closing_parenthesis(string, s)]
        name = problem_string.split(" ")[1]

        s = string.find(DOMAIN)
        domain_string = string[s:find_closing_parenthesis(string, s)]
        domain_name = domain_string.split(" ")[1]
        if domain_name != domain.name:
            print("Wrong domain name")

        s = string.find(OBJECTS)
        objects_string = string[s:find_closing_parenthesis(string, s)]
        objects = parse_objects(objects_string, domain.types)

        s = string.find(INIT)
        init_string = string[s:find_closing_parenthesis(string, s)]
        init = parse_init(init_string, domain.predicates, objects)

        s = string.find(GOAL)
        goal_string = string[s:find_closing_parenthesis(string, s)]
        goal = parse_goal(goal_string, domain.predicates, objects)

        return Problem(name, domain, objects, init, goal)

    def init_costs(self, robot_type: Type = None, human_type: Type = None):
        if not self.domain.has_costs():
            return
        self.init.append(Equals(Object("total-cost", Type("function", [])), Constant(0)))
        if self.domain.cost_mode == CostMode.MIN_HUMAN:
            robot_cost = 1
            human_cost = 100
        elif self.domain.cost_mode == CostMode.MIN_ROBOT:
            robot_cost = 100
            human_cost = 1
        elif self.domain.cost_mode == CostMode.BALANCED:
            robot_cost = 1
            human_cost = 1
        else:
            return

        for o in self.objects:
            if robot_type in o.type.get_all_supertypes() or o.type == robot_type:
                self.init.append(Equals(Object("(cost " + o.name + ")", robot_type), Constant(robot_cost)))
            elif human_type in o.type.get_all_supertypes() or o.type == human_type:
                self.init.append(Equals(Object("(cost " + o.name + ")", robot_type), Constant(human_cost)))

    def init_capacity(self, robot_type: Type = None, human_type: Type = None):
        robot_capacity = 1
        human_capacity = 3

        for o in self.objects:
            if robot_type in o.type.get_all_supertypes() or o.type == robot_type:
                self.init.append(
                    Equals(Object("(carry-capacity " + o.name + ")", robot_type), Constant(robot_capacity)))
            elif human_type in o.type.get_all_supertypes() or o.type == human_type:
                self.init.append(
                    Equals(Object("(carry-capacity " + o.name + ")", robot_type), Constant(human_capacity)))


def parse_objects(string: str, types: List[Type]) -> List[Object]:
    # TODO add error if type not in types list
    lines = string.split('\n')
    lines = list(filter(lambda x: "-" in x, lines))

    return [Object(l.split(" - ")[0].replace(" ", ""), get_type(l.split(" - ")[1], types)) for l in lines]


def parse_init(string: str, predicates: List[Predicate], objects: List[Object]) -> List[Predicate]:
    i = string.find("(")
    init = []

    while i >= 0:
        end = find_closing_parenthesis(string, i)
        predicate_string = string[i + 1:end]
        init.append(parse_predicate(predicate_string, predicates, objects))
        i = string.find("(", end)

    return init


def parse_literal(string: str, objects: List[Literal]) -> Literal:
    string = string.replace("?", "")
    if string.isdigit():
        return Constant(int(string))
    return get_object(string, objects)


def parse_predicate(string: str, predicates: List[Predicate], objects: List[Literal]) -> Predicate:
    tokens = string.split(" ")
    predicate_name = tokens[0].replace(" ", "")
    object_names = list(filter(lambda x: len(x) != 0, tokens[1:]))
    # remove ? to ease parsing of variables
    object_names = [name.replace("?", "") for name in object_names]
    objects = [parse_literal(name, objects) for name in object_names]
    predicate = get_predicate(predicate_name, predicates)
    if predicate:
        return predicate(objects).in_definition(False)
    else:
        raise PredicateParsingException(predicate_name, [str(p) for p in predicates])


def parse_goal(string: str, predicates: List[Predicate], objects: List[Literal]) -> LogicElement:
    string = string.replace("\n", "")
    i = string.find("(")
    end = find_closing_parenthesis(string, i)
    goal = parse_logic_element(string[i + 1:end], predicates, objects)
    # the goal cannot be only one opredicate it needs to be embedded withi an narry logic expression
    if isinstance(goal, Predicate) or isinstance(goal, Not):
        return And([goal])
    return goal


def parse_logic_element(string: str, predicates: List[Predicate], objects: List[Literal]) -> LogicElement:
    function_keywords = ("=", "increase", "decrease")
    if string.startswith(('and', "or", "not")):
        i = string.find("(")
        logic_elements = []
        while i >= 0:
            end = find_closing_parenthesis(string, i)
            logic_element_string = string[i + 1:end]
            logic_elements.append(parse_logic_element(logic_element_string, predicates, objects))
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
            elem = parse_logic_element(logic_element_string, predicates, objects)
            if i == 0:
                first = elem
            elif end == len(arguments_string) - 1:
                second = elem
            i = arguments_string.find("(", end)
        args = [a for a in arguments_string.split(" ") if a != ""]
        if first is None:
            first = parse_literal(args[0], objects)
        if second is None:
            second = parse_literal(args[-1], objects)
        if keyword == "=":
            return Equals(first, second)
        elif keyword == "increase":
            return Increase(first, second)
        elif keyword == "decrease":
            return Decrease(first, second)
    else:
        return parse_predicate(string, predicates, objects)


def get_type(name: str, types: List[Type]) -> Type:
    return next((type for type in types if type.name == name), None)


def get_predicate(name: str, predicates: List[Predicate]) -> Predicate:
    return next((predicate for predicate in predicates if predicate.name == name), None)


def get_object(name: str, objects: List[Literal]) -> Literal:
    object = next((object for object in objects if object.name == name), None)
    if object:
        return object
    else:
        raise ObjectParsingException(name, [o.name for o in objects])
