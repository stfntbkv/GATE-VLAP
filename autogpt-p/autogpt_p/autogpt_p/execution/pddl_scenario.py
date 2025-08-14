from typing import List

from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceMappingDatabase
from object_detection.detection_memory_segment import DetectedObject, ObjectRelation

from autogpt_p.execution.actor_skill_mapping import ActorSkillMapping
from pddl.core import Requirements, Type, Variable, Predicate, DerivedPredicate, \
    Action, ExistsOp, Not, And, Equals, Or, Object, CostMode, LogicOp, Constant, Decrease, Increase
from pddl.domain import Domain
from pddl.problem import Problem, parse_goal


def define_generic_object_list(length: int, object_type):
    return [Variable("o" + str(i), object_type) for i in range(length)]


ACTOR_SUFFIX = "er"
ACTOR_VARIABLE = "a"

OBJECT_TYPE = Type("object", [])
LOCATION_TYPE = Type("location", [OBJECT_TYPE])
ACTOR_TYPE = Type("actor", [LOCATION_TYPE])
ROBOT_TYPE = Type("robot", [ACTOR_TYPE])
HUMAN_TYPE = Type("human", [ACTOR_TYPE])
CARRY_TYPE = Type("carry", [OBJECT_TYPE])
SUPPORT_TYPE = Type("support", [OBJECT_TYPE])
CONTAIN_TYPE = Type("contain", [OBJECT_TYPE])
ASSISTED_CARRY_TYPE = Type("assisted-carry", [OBJECT_TYPE])
LIQUID_CONTAIN_TYPE = Type("liquid-contain", [OBJECT_TYPE])
LIQUID_TYPE = Type("liquid", [OBJECT_TYPE])
OPEN_TYPE = Type("open", [OBJECT_TYPE])
CLOSE_TYPE = Type("close", [OBJECT_TYPE])
WET_SWIPE_TYPE = Type("wet-swipe", [OBJECT_TYPE])
CONSUMABLE_TYPE = Type("consumable", [OBJECT_TYPE])
DUMMY_TYPE = Type("dummy", [OBJECT_TYPE])

ON_COMMENT = "Describes {} being on top of the supporting {}"
IN_COMMENT = "Describes the non liquid {} being in the container {}"
LIQUID_IN_COMMENT = "Describes the liquid {} being in the container {}"
AT_COMMENT = "Describes the actor {} being at the location {} which can also be another actor"
INHAND_COMMENT = "Describes the object {} being in the hand of a human or robot {}. " \
                 "A human can have multiple objects in their hand"
CARRIED_COMMENT = "Describes the object {} being carried together by the actors {} and {}"
FREE_COMMENT = "Describes that there is no object on top of the supporting {}"
OPENED_COMMENT = "Describes that {} is open"
CLOSED_COMMENT = "Describes that {} is closed"
CHOPPED_COMMENT = "Describes that a consumable {} is chopped"
CLEAN_COMMENT = "Describes that an object {} has been cleaned with water"
VISITED_COMMENT = "Describes that the robot has visited the given location {}"

# generic predicates without defined types
ON = Predicate("on", define_generic_object_list(2, OBJECT_TYPE), )
IN = Predicate("in", define_generic_object_list(2, OBJECT_TYPE))
LIQUID_IN = Predicate("liquid_in", define_generic_object_list(2, OBJECT_TYPE))
AT = Predicate("at", define_generic_object_list(2, OBJECT_TYPE))
INHAND = Predicate("inhand", define_generic_object_list(2, OBJECT_TYPE))
CARRIED = Predicate("carried", define_generic_object_list(3, OBJECT_TYPE))
HAND_OCCUPIED = Predicate("hand_occupied", define_generic_object_list(1, OBJECT_TYPE))
INDIRECT_ON = Predicate("indirect_on", define_generic_object_list(2, OBJECT_TYPE))
REACHABLE = Predicate("reachable", define_generic_object_list(3, OBJECT_TYPE))
EMPTY = Predicate("empty", define_generic_object_list(1, OBJECT_TYPE))
FREE = Predicate("free", define_generic_object_list(1, OBJECT_TYPE))
OPENED = Predicate("opened", define_generic_object_list(1, OBJECT_TYPE))
CLOSED = Predicate("closed", define_generic_object_list(1, OBJECT_TYPE))
TURN_ON = Predicate("turn_on", define_generic_object_list(1, OBJECT_TYPE))
TURN_OFF = Predicate("turn_off", define_generic_object_list(1, OBJECT_TYPE))
CHOPPED = Predicate("chopped", define_generic_object_list(1, OBJECT_TYPE))
LIQUID_WARM = Predicate("liquid_warm", define_generic_object_list(1, OBJECT_TYPE))
WET = Predicate("wet", define_generic_object_list(1, OBJECT_TYPE))
CLEAN = Predicate("clean", define_generic_object_list(1, OBJECT_TYPE))
VISITED = Predicate("visited", define_generic_object_list(1, OBJECT_TYPE))

# cooking domain
STIRRED = Predicate("stirred", define_generic_object_list(1, OBJECT_TYPE))
MIXED = Predicate("mixed", define_generic_object_list(2, OBJECT_TYPE))
COOKED = Predicate("cooked", define_generic_object_list(1, OBJECT_TYPE))


CARRY_CAPACITY = Predicate("carry-capacity", define_generic_object_list(1, OBJECT_TYPE))


def affordance_mapping2pddl_types(affordance_mapping: ObjectAffordanceMappingDatabase, detected_objects=None):
    types = [Type(aff.name, [OBJECT_TYPE]) for aff in affordance_mapping.oa_database.affordances]
    types.append(LOCATION_TYPE)
    types.append(OBJECT_TYPE)
    types.append(ACTOR_TYPE)
    types.append(ROBOT_TYPE)
    types.append(HUMAN_TYPE)
    types.append(DUMMY_TYPE)

    aff_dict = {type.name: type for type in types}

    objects = [Type(format_object_class(oam.object_class.name), [aff_dict[aff.name] for aff in oam.affordances]) for oam
               in affordance_mapping.oams]

    if detected_objects:
        detected_types = [o.class_name for o in detected_objects]
        objects = filter(lambda type: type.name in detected_types, objects)

    types.extend(objects)
    aff_dict = {type.name: type for type in types}

    return types, aff_dict


def define_domain(domain_name: str, affordance_mapping: ObjectAffordanceMappingDatabase, detected_objects=None,
                  actor_skill_mapping: ActorSkillMapping = None, generic_predicates=True, cost_mode=CostMode.MIN_HUMAN):
    requirements = [Requirements.STRIPS, Requirements.TYPING, Requirements.NEG_PRECONDITION,
                    Requirements.EXISTENTIAL_PRECONDITION, Requirements.EQUALITY,
                    Requirements.DERIVED_PREDICATES, Requirements.DIS_PRECONDITION]
    types, aff_dict = affordance_mapping2pddl_types(affordance_mapping, detected_objects)

    # define variables
    o = Variable("o", aff_dict["carry"])
    g = Variable("g", aff_dict["grasp"])
    os = Variable("os", aff_dict["carry"])
    ca = Variable("ca", aff_dict["carry"])
    s = Variable("s", aff_dict["support"])
    ins = Variable("ins", aff_dict["support"])
    c = Variable("c", aff_dict["contain"])
    tc = Variable("tc", aff_dict["contain"])
    ac = Variable("ac", aff_dict["assisted-carry"])
    lc = Variable("lc", aff_dict["liquid-contain"])
    tlc = Variable("t", aff_dict["liquid-contain"])
    a = Variable("a", aff_dict["robot"])
    r = Variable("r", aff_dict["actor"])
    actor = Variable("actor", aff_dict["actor"])
    other = Variable("other", aff_dict["actor"])
    l = Variable("l", aff_dict["location"])
    li = Variable("li", aff_dict["liquid"])
    d = Variable("d", aff_dict["location"])
    op = Variable("op", aff_dict["open"])
    cl = Variable("cl", aff_dict["close"])
    ph = Variable("ph", aff_dict["push"])
    pl = Variable("pl", aff_dict["pull"])
    ws = Variable("ws", aff_dict["wet-swipe"])
    p = Variable("p", aff_dict["pour"])
    cut = Variable("cut", aff_dict["cut"])
    co = Variable("co", aff_dict["consumable"])
    h = Variable("h", aff_dict["heat"])
    hr = Variable("hr", aff_dict["heat-resistance"])
    ss = Variable("ss", aff_dict["sturdy-support"])
    wa = Variable("wa", Type("water", [aff_dict["liquid"], aff_dict["drinkable"]]))

    # it does not make sense to restrict the types of predicates as the types are restricted by the actions enough
    # there cannot be an "illegal" predicate if the start state is legal and the conditions of the actions are legal
    if generic_predicates:
        predicates = [ON, IN, LIQUID_IN, AT, INHAND, CARRIED, EMPTY, INDIRECT_ON, REACHABLE, FREE,
                      OPENED, CLOSED, TURN_ON, TURN_OFF, LIQUID_WARM, WET, CHOPPED, CLEAN, HAND_OCCUPIED, VISITED]
    else:
        predicates = define_non_generic_predicates()

    definition_predicates = [p.in_definition() for p in predicates]

    # define derived predicates
    go = define_generic_object_list(1, OBJECT_TYPE)[0]
    indirect_on_derived = DerivedPredicate(INDIRECT_ON([o, s]),
                                           ExistsOp([go], And([ON([o, go]),
                                                               Or([INDIRECT_ON([go, s]), ON([go, s])])])))
    reachable_derived = DerivedPredicate(REACHABLE([o, l, a]),
                                         And([AT([a, l]),
                                              Or([Equals(l, o), ON([o, l]), INDIRECT_ON([o, l]), AT([o, l])])]))
    free_derived = DerivedPredicate(FREE([o]), Not(ExistsOp([os], ON([os, o]))))
    hand_occupied_derived = DerivedPredicate(HAND_OCCUPIED([a]), Equals(CARRY_CAPACITY([a]), Constant(0)))
    derived_predicates = [indirect_on_derived, reachable_derived, free_derived]

    make_at_symmetric = Action("make-at-symmetric",
                               [actor, other],
                               And([AT([actor, other])]),
                               And([AT([other, actor])]),
                               [])
    make_at_symmetric = Action("make-at-symmetric",
                               [actor, other],
                               And([AT([actor, other])]),
                               And([AT([other, actor])]),
                               [])

    if not actor_skill_mapping:
        # define actions
        grasp = Action("GRASP",
                       [a, o, s, l],
                       And([Not(HAND_OCCUPIED([a])), FREE([o]), ON([o, s]), REACHABLE([s, l, a])]),
                       And([INHAND([o, a]), Decrease(CARRY_CAPACITY([a]), Constant(1)), Not(ON([o, s]))]))
        # these are inverse actions v ^
        place = Action("PLACE",
                       [a, o, s, l],
                       And([INHAND([o, a]), REACHABLE([s, l, a])]),
                       And([ON([o, s]), Not(INHAND([o, a])), Not(Equals(CARRY_CAPACITY([a]), Constant(0)))]))

        take_out = Action("TAKE_OUT",
                          [a, o, c, l],
                          And([Not(HAND_OCCUPIED([a])), FREE([o]), IN([o, c]), Not(CLOSED([c])), REACHABLE([c, l, a])]),
                          And([INHAND([o, a]), Decrease(CARRY_CAPACITY([a]), Constant(1)), Not(IN([o, c]))]))
        # these are inverse actions v ^
        putin = Action("PUTIN",
                       [a, o, c, l],
                       And([INHAND([o, a]), Not(CLOSED([c])), REACHABLE([c, l, a])]),
                       And([IN([o, c]), Not(INHAND([o, a])), Not(Equals(CARRY_CAPACITY([a]), Constant(0)))]))
        handover = Action("HANDOVER",
                          [a, r, o],
                          And([INHAND([o, a]), AT([a, r]), Not(HAND_OCCUPIED([a]))]),
                          And([Not(INHAND([o, a])), Decrease(CARRY_CAPACITY([a]), Constant(1)), INHAND([o, r]),
                               Increase(CARRY_CAPACITY([a]), Constant(1))]),
                          [0, 1])

        # move is its own inverse action
        move = Action("MOVE",
                      [a, l, d],
                      And([AT([a, l])]),
                      And([Not(AT([a, l])), AT([a, d])]))

        pour = Action("POUR",
                      [a, p, o, tc, l],
                      And([IN([o, p]), INHAND([p, a]), REACHABLE([tc, l, a]), Not(CLOSED([tc]))]),
                      And([IN([o, tc])]))
        # these two are equivalent but fill is pour with liquids
        fill = Action("FILL",
                      [a, lc, li, tlc, l],
                      And([LIQUID_IN([li, lc]), INHAND([lc, a]), REACHABLE([tlc, l, a]), Not(CLOSED([lc])),
                           Not(CLOSED([tlc]))]),
                      And([LIQUID_IN([li, tlc])]))
        open = Action("OPEN",
                      [a, op, l],
                      And([CLOSED([op]), REACHABLE([op, l, a])]),
                      And([Not(CLOSED([op])), OPENED([op])]))
        close = Action("CLOSE",
                       [a, cl, l],
                       And([OPENED([cl]), REACHABLE([cl, l, a])]),
                       And([Not(OPENED([cl])), CLOSED([cl])]))

        # Generic turn on/off actions for any object
        ob = Variable("ob", OBJECT_TYPE)
        turn_on = Action("TURN_ON",
                        [a, ob, l],
                        And([Not(TURN_ON([ob])), REACHABLE([ob, l, a]), Not(HAND_OCCUPIED([a]))]),
                        And([TURN_ON([ob]), Not(TURN_OFF([ob]))]))
        turn_off = Action("TURN_OFF",
                         [a, ob, l],
                         And([TURN_ON([ob]), REACHABLE([ob, l, a]), Not(HAND_OCCUPIED([a]))]),
                         And([TURN_OFF([ob]), Not(TURN_ON([ob]))]))

        # the stack operation needs to be dynamically created to ensure that only objects of the same type can be stacked

        chop = Action("CHOP",
                      [a, cut, co, ss, l],
                      And([INHAND([cut, a]), REACHABLE([co, l, a]), ON([co, ss])]),
                      And([CHOPPED([co])]))

        heat_liquid = Action("HEAT_LIQUID",
                             [a, lc, hr, li, h, s, l],
                             Or([
                                 And([LIQUID_IN([li, lc]), Equals(lc, hr), ON([lc, s]), Equals(s, h)]),
                                 And([LIQUID_IN([li, lc]), Equals(lc, h)])]),
                             And([LIQUID_WARM([li])]))

        # this is more an effect than an action, just give this cost 0
        wetten = Action("wetten",
                        [ws, c, lc, wa],
                        And([Not(WET([ws])), IN([ws, c]), LIQUID_IN([wa, lc]), Equals(c, lc)]),
                        And([WET([ws])]))
        wet_swipe = Action("WET_SWIPE",
                           [a, ws, o, l],
                           And([WET([ws]), INHAND([ws, a]), REACHABLE([o, l, a])]),
                           And([CLEAN([o])]))

        actions = [grasp, place, take_out, putin, handover, move, pour, fill, open, close, turn_on, turn_off, chop, heat_liquid, wetten,
                   wet_swipe, make_at_symmetric]
    else:
        actions = actor_skill_mapping.make_pddl_actions(ACTOR_TYPE, ACTOR_TYPE, predicates)
        actions.append(make_at_symmetric)
        dummy = Action("dummy",
                       [Variable("dummy", DUMMY_TYPE), p, o, tc],
                       And([IN([o, p])]),
                       And([IN([o, tc])]))
        wetten = Action("wetten",
                        [ws, c, lc, wa],
                        And([Not(WET([ws])), IN([ws, c]), LIQUID_IN([wa, lc]), Equals(c, lc)]),
                        And([WET([ws])]),[])
        
        # Add turn on/off actions to actor skill mapping path as well
        ob = Variable("ob", OBJECT_TYPE)
        actor = Variable("actor", ACTOR_TYPE) 
        l = Variable("l", LOCATION_TYPE)
        turn_on_asm = Action("TURN_ON",
                            [actor, ob, l],
                            And([Not(TURN_ON([ob])), REACHABLE([ob, l, actor]), Not(HAND_OCCUPIED([actor]))]),
                            And([TURN_ON([ob]), Not(TURN_OFF([ob]))]))
        turn_off_asm = Action("TURN_OFF",
                             [actor, ob, l],
                             And([TURN_ON([ob]), REACHABLE([ob, l, actor]), Not(HAND_OCCUPIED([actor]))]),
                             And([TURN_OFF([ob]), Not(TURN_ON([ob]))]))
        
        actions.append(dummy)
        actions.append(wetten)
        actions.append(turn_on_asm)
        actions.append(turn_off_asm)
        types.extend(actor_skill_mapping.get_action_types(ACTOR_TYPE))
        types.extend(actor_skill_mapping.get_profile_types(ACTOR_TYPE, ROBOT_TYPE, HUMAN_TYPE))

    # add stacking actions as they have to be dynamically created based on stackable objects
    stackable = [type for type in aff_dict.values() if type.is_subtype(aff_dict["stack"])]
    # stack_actions = define_stacking_actions(a, l, stackable)
    # actions += stack_actions

    domain = Domain(domain_name, requirements, types, definition_predicates, derived_predicates, actions, cost_mode,
                    ACTOR_TYPE)
    # print(domain)
    return domain


def make_at_reflexive(goal: LogicOp) -> LogicOp:
    # TODO maybe use actions for this
    for element in goal.logic_elements:
        if isinstance(element, Predicate):
            if element.name == "at" and element.variables:
                return goal
        elif isinstance(element, LogicOp):
            make_at_reflexive(element)
    return goal


def define_problem(name, domain, detected_objects, detected_relations, locations,
                   actor_skill_mapping: ActorSkillMapping,
                   starting_locations=None, goal=""):
    starting_locations = starting_locations if starting_locations else []
    profile_types = actor_skill_mapping.get_profile_types(ACTOR_TYPE, ROBOT_TYPE, HUMAN_TYPE)

    actors = actor_skill_mapping.get_actor_objects(profile_types)

    objects = [Object(format_object(o), list(filter(lambda type: o.class_name == type.name, domain.types))[0])
               for o in detected_objects]

    for object in objects:
        if object.name in [a.name for a in actors]:
            objects.remove(object)

    location_names = locations
    for o in objects:
        if o.name in location_names:
            domain.types[domain.types.index(o.type)].add_supertype(LOCATION_TYPE)

    objects += actors

    objects = list(set(objects))  # easy way to remove duplicates

    init = list(set(define_predicates(detected_relations)))
    init += [AT([actors[i], location_names[j]]) for i, j in enumerate(starting_locations)]

    goal = parse_goal(goal, domain.predicates, objects) if len(goal) > 0 else None

    problem = Problem(name, domain, objects, init, goal, ROBOT_TYPE, HUMAN_TYPE)
    return problem


def define_problem_goal(name, domain, detected_objects, detected_relations, locations,
                        actor_skill_mapping: ActorSkillMapping,
                        starting_locations=None, goal=None):
    starting_locations = starting_locations if starting_locations else []
    profile_types = actor_skill_mapping.get_profile_types(ACTOR_TYPE, ROBOT_TYPE, HUMAN_TYPE)

    actors = actor_skill_mapping.get_actor_objects(profile_types)

    objects = [Object(format_object(o), list(filter(lambda type: o.class_name == type.name, domain.types))[0])
               for o in detected_objects]

    for object in objects:
        if object.name in [a.name for a in actors]:
            objects.remove(object)

    location_names = locations
    for o in objects:
        if o.name in location_names:
            domain.types[domain.types.index(o.type)].add_supertype(LOCATION_TYPE)

    objects += actors

    objects = list(set(objects))  # easy way to remove duplicates

    init = list(set(define_predicates(detected_relations)))
    init += [AT([actors[i], location_names[j]]) for i, j in enumerate(starting_locations)]

    problem = Problem(name, domain, objects, init, goal, ROBOT_TYPE, HUMAN_TYPE)
    return problem


def define_stacking_actions(actor: Variable, location: Variable, stackable_classes: List[Type]) -> List[Action]:
    actions = []
    for stackable in stackable_classes:
        s1 = Variable("s1", stackable)
        s2 = Variable("s2", stackable)
        action = Action("STACK_" + str(stackable.name).upper(),
                        [actor, s1, s2, location],
                        And([REACHABLE([s2, location, actor]), INHAND([s1, actor])]),
                        And([ON([s1, s2]), Not(INHAND([s1, actor])),
                             Not(Equals(CARRY_CAPACITY([actor]), Constant(0)))]))
        actions.append(action)
    return actions


def define_predicates(detected_relations: List[ObjectRelation]) -> List[Predicate]:
    return [Predicate(r.relation_name, [Object(format_object(o), Type(format_object_class(o.class_name), []))
                                        for o in r.related_objects]) for r in detected_relations]


def define_actions(skills, generic_actor_type: Type):
    actions = []
    for skill in skills:
        name = skill.name
        parameters = skill.parameters
        actor_name = skill.name + ACTOR_SUFFIX
        actor_type = Type(actor_name, [generic_actor_type])
        actor_variable = Variable("a", actor_type)
        preconditions = skill.preconditions
        effects = skill.effects

        skill_action = Action(name, [actor_variable] + parameters, preconditions, effects)
        actions.append(skill_action)
    return actions


def define_actors(actors, generic_actor_type: Type) -> List[Type]:
    return [Type(actor.name, [Type(make_skill_type_name(skill), [generic_actor_type]) for skill in actor.skills])
            for actor in actors]


def make_skill_type_name(skill):
    return skill.name + ACTOR_SUFFIX


def format_object_class(object_class: str):
    return object_class.replace(" ", "_")


def format_object(detected_object: DetectedObject) -> str:
    return format_object_class(detected_object.class_name) + str(detected_object.id)


def define_non_generic_predicates():
    o = Variable("o", CARRY_TYPE)
    ob = Variable("ob", OBJECT_TYPE)
    s = Variable("s", SUPPORT_TYPE)
    c = Variable("c", CONTAIN_TYPE)
    ac = Variable("ac", ASSISTED_CARRY_TYPE)
    lc = Variable("lc", LIQUID_CONTAIN_TYPE)
    a = Variable("a", ACTOR_TYPE)
    r = Variable("r", ACTOR_TYPE)
    l = Variable("l", LOCATION_TYPE)
    li = Variable("li", LIQUID_TYPE)
    op = Variable("op", OPEN_TYPE)
    cl = Variable("cl", CLOSE_TYPE)
    ws = Variable("ws", WET_SWIPE_TYPE)
    co = Variable("co", CONSUMABLE_TYPE)

    on = Predicate("on", [o, s], ON_COMMENT)
    indirect_on = Predicate("indirect_on", [o, s], "Do not use this predicate")
    inside = Predicate("in", [o, c], IN_COMMENT)
    liquid_in = Predicate("liquid_in", [li, lc], LIQUID_IN_COMMENT)
    at = Predicate("at", [a, l], AT_COMMENT)
    inhand = Predicate("inhand", [o, a], INHAND_COMMENT)
    carried = Predicate("carried", [ac, a, r], CARRIED_COMMENT)
    hand_occupied = Predicate("hand_occupied", [a], "Do not use this predicate")
    free = Predicate("free", [s], "Do not use this predicate")
    opened = Predicate("opened", [op], OPENED_COMMENT)
    closed = Predicate("closed", [cl], CLOSED_COMMENT)
    turn_on = Predicate("turn_on", [ob], "Describes that {} is turned on")
    turn_off = Predicate("turn_off", [ob], "Describes that {} is turned off")
    warm = Predicate("warm", [co])
    liquid_warm = Predicate("liquid_warm", [li])
    reachable = Predicate("reachable", [o, l, a], "Do not use this predicate")
    wet = Predicate("wet", [ws])
    chopped = Predicate("chopped", [co], CHOPPED_COMMENT)
    clean = Predicate("clean", [ob], CLEAN_COMMENT)
    visited = Predicate("visited", [l], VISITED_COMMENT)

    return [on, indirect_on, inside, liquid_in, at, inhand, carried, free, reachable, opened, closed, turn_on, turn_off, warm,
            liquid_warm, wet, chopped, clean, visited, hand_occupied]
