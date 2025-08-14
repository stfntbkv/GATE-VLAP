import logging
from typing import Tuple, List
import numpy as np

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.execution.pddl_scenario import define_predicates, define_domain, define_problem_goal

from autogpt_p.helpers.dnf import goal_to_dnf
from autogpt_p.helpers.prolog_wrapper import PrologWrapper
from pddl.core import LogicElement, Predicate, Not
from autogpt_p.planning.autogpt_planner import AutoGPTPlanner
from autogpt_p.planning.planner import FastDownwardPlanner


def dnf_to_string(dnf: List[List[Predicate]]) -> str:
    return str([[str(pred) for pred in substate] for substate in dnf])


class LiberoSimulatedScene:
    """LIBERO-specific simulated scene representation"""
    
    def __init__(self, objects, relations, locations):
        self.objects = objects
        self.relations = relations
        self.locations = locations
    
    def get_locations(self):
        return self.locations


class LiberoPlannerEvaluationScenario:
    """
    LIBERO-specific evaluation scenario that handles LIBERO task characteristics
    """

    def __init__(self, scene: LiberoSimulatedScene, user_task: str, goal_state: LogicElement, min_costs: int):
        """
        Creates a new LIBERO evaluation scenario
        :param scene: LiberoSimulatedScene containing objects, relations, and locations
        :param user_task: the natural language description of what the user wants
        :param goal_state: the ideal goal state set by the evaluation designer
        :param min_costs: minimum cost to achieve the goal
        """
        self.scene = scene
        self.user_task = user_task
        self.goal_state = goal_state
        self.min_costs = min_costs

    def evaluate_planner(self, planner: AutoGPTPlanner) -> Tuple[bool, int, float, float, float, bool, float]:
        """
        Uses AutoGPTPlanner to generate a plan for the given LIBERO task and calculates metrics
        :param planner: the planner to use
        :return: a tuple of (success, loops, accuracy, precision, f1, is_min_cost, cost_rate)
        """
        logging.info("NEW LIBERO CASE --------------------------------------------------------")
        print(f"LIBERO Task: {self.user_task}")
        
        # Generate the plan
        planner.reset_history()
        planner.update_scene(self.scene.objects, self.scene.relations, self.scene.get_locations())
        plan = planner.plan(self.user_task)
        
        logging.info("----------------------------LIBERO Results:---------------------------")
        logging.info("Task: " + str(self.user_task))
        logging.info("Generated Plan:\n" + str(plan))

        # Find results of the plan using Prolog simulation
        import time
        start_time = time.time()
        prolog = PrologWrapper()
        generated_goal = planner.problem.goal
        logging.info("Generated Goal: " + str(generated_goal))
        logging.info("Desired Goal: " + str(self.goal_state))
        
        # Simulate plan execution
        resulting_state = prolog.get_result_from_plan(self.scene.objects, self.scene.relations, plan)
        sim_time = time.time()
        print(f"DEBUG: Prolog simulation took {sim_time - start_time:.2f} seconds")
        resulting_state = define_predicates(resulting_state)

        # Evaluate resulting state and goal
        success = self._matches_goal(resulting_state)
        end_time = time.time()
        print(f"DEBUG: Total LIBERO evaluation took {end_time - start_time:.2f} seconds")
        
        logging.info("Goal Reached: " + str(success))
        loops = planner.feedback_loops
        acc, prec, f1 = self._get_f1(generated_goal)
        is_min = (plan.get_real_length() == int(self.min_costs))
        logging.info("Costs are: {} -- minimal costs: {}".format(plan.get_real_length(), self.min_costs))
        logging.info("Costs minimal: " + str(is_min))
        cost_rate = plan.get_real_length() / float(self.min_costs) if float(self.min_costs) > 0.0 else plan.get_real_length()
        logging.info("Costs Rate: " + str(cost_rate))
        logging.info("------------------------------------------------------------------------")
        
        return success, loops, acc, prec, f1, is_min, cost_rate

    def validate(self, oam_db, asm):
        """Validate that the scenario is solvable using Fast Downward planner"""
        domain = define_domain("libero_robotic_planning", oam_db, self.scene.objects, asm, False)
        problem = define_problem_goal("libero_test", domain, self.scene.objects, self.scene.relations,
                                      self.scene.get_locations(), asm, goal=self.goal_state)
        self.feedback_loops = 0
        planner = FastDownwardPlanner(domain)
        plan = planner.solve(problem)
        print(f"Validation plan: {plan}")

    def _matches_goal(self, state: List[Predicate]) -> bool:
        """Check if the resulting state matches the desired goal"""
        dnf = goal_to_dnf(self.goal_state)
        print("DNF = " + dnf_to_string(dnf))

        for sub_state in dnf:
            positive, negative = self._get_pos_neg(sub_state)
            
            # Check that no negative predicates are true
            for predicate in negative:
                if predicate in state:
                    return False
            
            # Check that all positive predicates are true
            if set(positive).issubset(set(state)):
                return True
        
        return False

    def _get_pos_neg(self, substate: List[LogicElement]):
        """Separate positive and negative predicates from a goal substate"""
        positive = [s for s in substate if isinstance(s, Predicate)]
        negative = [s.logic_elements[0] for s in substate if isinstance(s, Not)]
        return positive, negative

    def _matching_rate(self, state: List[Predicate]):
        """Calculate matching rate between state and goal"""
        dnf = goal_to_dnf(self.goal_state)
        return max([matching_rate(sub_state, state) for sub_state in dnf])

    def _get_f1(self, generated_goal: LogicElement):
        """Calculate F1 score between generated and desired goals"""
        if generated_goal is None:
            return 0.0, 0.0, 0.0
        
        dnf = goal_to_dnf(self.goal_state)
        dnf_generated = goal_to_dnf(generated_goal)
        
        # Create matrices comparing all combinations of goal clauses
        recall_matrix = [[matching_rate(a, b) for a in dnf_generated] for b in dnf]
        precision_matrix = [[matching_rate(b, a) for a in dnf_generated] for b in dnf]
        
        print(f"Recall matrix: {recall_matrix}")
        print(f"Precision matrix: {precision_matrix}")
        
        # Calculate recall and precision
        recall = float(np.mean([max(column) for column in zip(*recall_matrix)]))
        precision = float(np.mean([max(column) for column in zip(*precision_matrix)]))
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        
        return precision, recall, f1


def jaccard_similarity(set_a: set, set_b: set):
    """
    Returns the jaccard similarity between two sets |A ∩ B| / |A ∪ B|
    :param set_a: the first set
    :param set_b: the second set
    :return: the jaccard similarity between 0 and 1
    """
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if len(union) != 0 else 0


def matching_rate(a: List[Predicate], b: List[Predicate]) -> float:
    """
    Returns the percentage of elements in a that are also in b
    :param a: the first set
    :param b: the second set
    :return: the percentage of elements in a that are also in b
    """
    common = set(a).intersection(set(b))
    return len(common) / len(a) if len(a) != 0 else 0