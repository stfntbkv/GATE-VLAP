
from typing import List
import signal
import os

from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceMappingDatabase
from object_detection.detection_memory_segment import DetectedObject, ObjectRelation

from autogpt_p.incremental_goal_memory.incremental_goal_memory import IncrementalGoalMemory
from autogpt_p.llm.llm_interface import NoGoalException, LLMInterface
from autogpt_p.execution.pddl_scenario import define_domain, define_problem
from pddl.core import Predicate
from pddl.plan import Plan
from autogpt_p.planning.goal_validator import MissingGoal, UnknownPredicate, UnknownObject, GoalValidator, \
    PredicateLimitation, TypingError
from pddl.problem import PredicateParsingException, ObjectParsingException
from autogpt_p.planning.planner import FastDownwardPlanner
from autogpt_p.planning.validation_error_handler import ValidationErrorHandler

def _timeout_function(f, timeout, *args, **kwargs):
    # Set the signal handler and a timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = f(*args, **kwargs)
    finally:
        signal.alarm(0)  # Disable the alarm

    return result


def get_limitations():
    l1 = PredicateLimitation("an object can only be on one other object", "forall(on(X,Y), not(on_other(X,Y,Z)))")
    l2 = PredicateLimitation("an object can only be in one other object", "forall(in(X,Y), (not(in_other(X,Y,Z)), "
                                                                          "not(on_other(X,Y,Z))))")
    l3 = PredicateLimitation("an object that is in the hand of an actor cannot be in another hand or at another "
                             "place", "forall(inhand(X,Y), (not(in_other_hand(X,Y,Z)), not(on_other(X,Y,Z)),  "
                                      "not(in_other(X,Y,Z))))")
    l4 = PredicateLimitation("an actor can only be at one location", "forall(at(X,Y), not(at_other(X,Y,Z)))")
    l5 = PredicateLimitation("a liquid can only be in one other container", "forall(liquid_in(X,Y), "
                                                                            "not(liquid_in_other(X,Y,Z)))")
    return [l1, l2, l3, l4, l5]


class AutoGPTPlanner:
    """
    This class represents the inner feedback loop used for planning, based on a user instruction
    """

    def __init__(self, chatgpt: LLMInterface, objects: List[DetectedObject], relations: List[ObjectRelation],
                 locations: List[str], oam_db: ObjectAffordanceMappingDatabase, actor_skill_mapping=None,
                 partial_plan=False, max_predicates=5, max_loops=5, number_of_examples=3, save_pddl_files=False):

        self.actor_skill_mapping = actor_skill_mapping
        self.problem = None
        self.domain = None
        self.relations = None
        self.objects = None
        self.recent_plan = None
        self.locations = locations
        self.chatgpt = chatgpt
        self.chatgpt_fresh = self.chatgpt.branch()
        self.oam_db = oam_db
        self.update_scene(objects, relations, locations)
        self.max_loops = max_loops
        self.max_predicates = max_predicates
        self.partial_plan = partial_plan
        self.feedback_loops = 0
        self.number_of_examples = number_of_examples
        self.save_pddl_files = save_pddl_files

    def update_scene(self, objects: List[DetectedObject], relations: List[ObjectRelation], locations=None):
        self.locations = locations if locations else self.locations
        self.objects = objects
        self.relations = relations
        robot_names = [r.get_name() for r in self.actor_skill_mapping.get_robot_actors()] if self.actor_skill_mapping else []
        human_names = [h.get_name() for h in self.actor_skill_mapping.get_human_actors()] if self.actor_skill_mapping else []
        self.domain = define_domain("robotic_planning", self.oam_db, self.objects, self.actor_skill_mapping, False)
        self.problem = define_problem("test", self.domain, objects, relations, self.locations, self.actor_skill_mapping)
        self.feedback_loops = 0

    def reset_history(self):
        copy = self.chatgpt_fresh.branch()
        self.chatgpt = self.chatgpt_fresh
        self.chatgpt_fresh = copy

    def plan_with_incremental_goal_memory(self, user_task: str, number_of_examples: int):
        # ask ChatGPT for goal
        found_goal = False
        error_handler = None
        plan = Plan([])
        plan.costs = -1
        # print(self.locations)
        predicates = [Predicate(p.name, p.variables, p.comment, p.definition) for p in self.domain.predicates]
        # this is the inner feedback loop responsible for getting a plan_with_incremental_goal_memory from a user instruction
        while self.feedback_loops < self.max_loops and not found_goal:
            self._make_generic(False)
            self.feedback_loops += 1
            if not error_handler:
                try:
                    if self.partial_plan:
                        goal_string = self.chatgpt.ask_for_partial_goal_in_context_learning(user_task, self.domain, self.problem,
                                                                        self.max_predicates, number_of_examples)
                    else:
                        goal_string = self.chatgpt.ask_for_goal_in_context_learning(user_task, self.domain, self.problem, number_of_examples)
                    print("GOAL:" + goal_string)
                except NoGoalException:
                    error_handler = ValidationErrorHandler(MissingGoal(), self.chatgpt)
                    print(error_handler.validation_error.error_message)
                    continue
            else:
                try:
                    goal_string = error_handler.correct_error()
                except NoGoalException:
                    error_handler = ValidationErrorHandler(MissingGoal(), self.chatgpt)
                    print(error_handler.validation_error.error_message)
                    continue
            self._make_generic(True)
            robot_names = [r.get_name() for r in self.actor_skill_mapping.get_robot_actors()] if self.actor_skill_mapping else []
            human_names = [h.get_name() for h in self.actor_skill_mapping.get_human_actors()] if self.actor_skill_mapping else []
            # construct problem
            if "?" in goal_string:
                error_handler = ValidationErrorHandler(TypingError(), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue
            try:
                # for planning using fast downward it is better to have generic predicates than typed predicates

                problem = define_problem("test", self.domain, self.objects, self.relations, self.locations,
                                         self.actor_skill_mapping, [], goal_string)
            except PredicateParsingException as e:
                error_handler = ValidationErrorHandler(UnknownPredicate(e.predicate_name), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue
            except ObjectParsingException as e:
                error_handler = ValidationErrorHandler(UnknownObject(e.object_name), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue

            # check for contradiction
            limitations = get_limitations()
            validator = GoalValidator(limitations, predicates, problem.goal)
            print(f"DEBUG: Validating goal with {len(limitations)} limitations")
            print(f"DEBUG: Goal to validate: {problem.goal}")
            print(f"DEBUG: Domain predicates: {[p.name for p in predicates]}")
            error = validator.validate()
            if not error:
                print("DEBUG: Goal validation PASSED - no contradictions found")
                print("DEBUG: Creating FastDownwardPlanner instance")
                planner = FastDownwardPlanner(self.domain)
                print(f"DEBUG: FastDownwardPlanner created with domain: {self.domain.name}")
                print(f"DEBUG: Domain has {len(self.domain.actions)} actions and {len(self.domain.predicates)} predicates")
                
                # DEBUG: Optionally save PDDL domain and problem to files
                if self.save_pddl_files:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # Use relative path for debug files
                    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    domain_file = os.path.join(script_dir, f"pddl_debug_files/debug_domain_{timestamp}.pddl")
                    problem_file = os.path.join(script_dir, f"pddl_debug_files/debug_problem_{timestamp}.pddl")
                    
                    # Ensure debug directory exists
                    os.makedirs(os.path.dirname(domain_file), exist_ok=True)
                    
                    with open(domain_file, 'w') as f:
                        f.write(str(self.domain))
                    with open(problem_file, 'w') as f:
                        f.write(str(problem))
                        
                    print(f"DEBUG: Saved domain to {domain_file}")
                    print(f"DEBUG: Saved problem to {problem_file}")
                
                print(f"DEBUG: Goal being planned: {problem.goal}")
                print(f"DEBUG: Problem objects: {[obj.name for obj in problem.objects]}")
                print("DEBUG: About to invoke FastDownward planner...")
                try:
                    plan = _timeout_function(planner.solve, 120, problem)
                    found_goal = True
                    self.problem = problem
                    print(f"DEBUG: Plan found successfully with cost {plan.costs}")
                except TimeoutError as e:
                    print("DEBUG: FastDownward planning timed out - plan too complicated, try partial plan_with_incremental_goal_memory")
                except Exception as e:
                    print(f"DEBUG: FastDownward planning failed with exception: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    if self.partial_plan:
                        self.max_predicates -= 1
                    else:
                        self.partial_plan = True

            else:
                print(f"DEBUG: Goal validation FAILED - error: {error.error_message}")
                error_handler = ValidationErrorHandler(error, self.chatgpt)

        self.recent_plan = plan

        return plan

    def plan_without_incremental_goal_memory(self, user_task: str):
        # ask ChatGPT for goal
        print(f"DEBUG: Starting plan_without_incremental_goal_memory for task: {user_task}")
        found_goal = False
        error_handler = None
        plan = Plan([])
        plan.costs = -1
        # print(self.locations)
        predicates = [Predicate(p.name, p.variables, p.comment, p.definition) for p in self.domain.predicates]
        # this is the inner feedback loop responsible for getting a plan_without_incremental_goal_memory from a user instruction
        while self.feedback_loops < self.max_loops and not found_goal:
            self._make_generic(False)
            self.feedback_loops += 1
            if not error_handler:
                try:
                    if self.partial_plan:
                        goal_string = self.chatgpt.ask_for_partial_goal(user_task, self.domain, self.problem,
                                                                        self.max_predicates)
                    else:
                        goal_string = self.chatgpt.ask_for_goal(user_task, self.domain, self.problem)  # hier autogpt in context learning aufrufen
                    print("GOAL:" + goal_string)
                except NoGoalException:
                    error_handler = ValidationErrorHandler(MissingGoal(), self.chatgpt)
                    print(error_handler.validation_error.error_message)
                    continue
            else:
                try:
                    goal_string = error_handler.correct_error()
                except NoGoalException:
                    error_handler = ValidationErrorHandler(MissingGoal(), self.chatgpt)
                    print(error_handler.validation_error.error_message)
                    continue
            self._make_generic(True)
            robot_names = [r.get_name() for r in self.actor_skill_mapping.get_robot_actors()] if self.actor_skill_mapping else []
            human_names = [h.get_name() for h in self.actor_skill_mapping.get_human_actors()] if self.actor_skill_mapping else []
            # construct problem
            if "?" in goal_string:
                error_handler = ValidationErrorHandler(TypingError(), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue
            try:
                # for planning using fast downward it is better to have generic predicates than typed predicates

                problem = define_problem("test", self.domain, self.objects, self.relations, self.locations,
                                         self.actor_skill_mapping, [], goal_string)
            except PredicateParsingException as e:
                error_handler = ValidationErrorHandler(UnknownPredicate(e.predicate_name), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue
            except ObjectParsingException as e:
                error_handler = ValidationErrorHandler(UnknownObject(e.object_name), self.chatgpt)
                print(error_handler.validation_error.error_message)
                continue

            # check for contradiction
            limitations = get_limitations()
            validator = GoalValidator(limitations, predicates, problem.goal)
            print(f"DEBUG: Validating goal with {len(limitations)} limitations")
            print(f"DEBUG: Goal to validate: {problem.goal}")
            print(f"DEBUG: Domain predicates: {[p.name for p in predicates]}")
            error = validator.validate()
            if not error:
                print("DEBUG: Goal validation PASSED - no contradictions found")
                print("DEBUG: Creating FastDownwardPlanner instance")
                planner = FastDownwardPlanner(self.domain)
                print(f"DEBUG: FastDownwardPlanner created with domain: {self.domain.name}")
                print(f"DEBUG: Domain has {len(self.domain.actions)} actions and {len(self.domain.predicates)} predicates")
                
                # DEBUG: Optionally save PDDL domain and problem to files
                if self.save_pddl_files:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    # Use relative path for debug files
                    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    domain_file = os.path.join(script_dir, f"pddl_debug_files/debug_domain_{timestamp}.pddl")
                    problem_file = os.path.join(script_dir, f"pddl_debug_files/debug_problem_{timestamp}.pddl")
                    
                    # Ensure debug directory exists
                    os.makedirs(os.path.dirname(domain_file), exist_ok=True)
                    
                    with open(domain_file, 'w') as f:
                        f.write(str(self.domain))
                    with open(problem_file, 'w') as f:
                        f.write(str(problem))
                        
                    print(f"DEBUG: Saved domain to {domain_file}")
                    print(f"DEBUG: Saved problem to {problem_file}")
                
                print(f"DEBUG: Goal being planned: {problem.goal}")
                print(f"DEBUG: Problem objects: {[obj.name for obj in problem.objects]}")
                print("DEBUG: About to invoke FastDownward planner...")
                try:
                    plan = _timeout_function(planner.solve, 120, problem)
                    found_goal = True
                    self.problem = problem
                    print(f"DEBUG: Plan found successfully with cost {plan.costs}")
                except TimeoutError as e:
                    print("DEBUG: FastDownward planning timed out - plan too complicated, try partial plan")
                except Exception as e:
                    print(f"DEBUG: FastDownward planning failed with exception: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    if self.partial_plan:
                        self.max_predicates -= 1
                    else:
                        self.partial_plan = True

            else:
                print(f"DEBUG: Goal validation FAILED - error: {error.error_message}")
                error_handler = ValidationErrorHandler(error, self.chatgpt)

        self.recent_plan = plan

        return plan

    def plan(self, user_task: str):
        """
        Wrapper method that calls the appropriate planning method based on configuration
        """
        print(f"DEBUG: plan() wrapper called for task: {user_task}")
        if hasattr(self, 'use_incremental_memory') and self.use_incremental_memory:
            print("DEBUG: Using plan_with_incremental_goal_memory")
            return self.plan_with_incremental_goal_memory(user_task, self.number_of_examples)
        else:
            print("DEBUG: Using plan_without_incremental_goal_memory")
            return self.plan_without_incremental_goal_memory(user_task)

    def _make_generic(self, generic: bool):
        self.domain = define_domain("robotic_planning", self.oam_db, self.objects, self.actor_skill_mapping, generic)
        # print("GENERIC DOMAIN")
        # print(str(self.domain))


def timeout_handler(signum, frame):
    raise TimeoutError("Function call timeout")
