import logging
from typing import Tuple, List
import numpy as np

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.execution.pddl_scenario import define_predicates, define_domain, define_problem_goal

from autogpt_p.helpers.dnf import goal_to_dnf
from autogpt_p.helpers.prolog_wrapper import PrologWrapper
from pddl.core import LogicElement, Predicate, Not

from autogpt_p.incremental_goal_memory.incremental_goal_memory import IncrementalGoalMemory
from autogpt_p.planning.autogpt_planner import AutoGPTPlanner
from autogpt_p.planning.planner import FastDownwardPlanner


def dnf_to_string(dnf: List[List[Predicate]]) -> str:
    return str([[str(pred) for pred in substate] for substate in dnf])


class IncrementalGoalMemoryEvaluationScenario:
    """

    """

    def __init__(self, scene: SimulatedScene, user_task: str, goal_state: LogicElement):
        """
        Creates a new evaluation scenario
        :param user_task: the natural language description of what the user wants
        :param goal_state: the ideal goal state set by the evaluation designer of what the user wants with his task
        """
        self.scene = scene
        self.user_task = user_task
        self.goal_state = goal_state

    def evaluate_incremental_goal_memory_without_memory(self, planner: AutoGPTPlanner) -> Tuple[bool, int, float, float, float, bool, float]:
        """
        Uses AutoGPTPlanner to generate a plan_with_incremental_goal_memory for the given user_task of this scenario and calculates different metrics
        :param planner: the planner that use
        :return: a tuple of whether the goal is sufficient, how many loops the planner took,
        the precision, the recall and the f1 score of the generated goal with the desired goal
        """
        logging.info("NEW CASE ------------------------------------------------------------------")
        print(self.user_task)
        # generate the plan_with_incremental_goal_memory
        planner.reset_history()
        planner.update_scene(self.scene.objects, self.scene.relations, self.scene.get_locations())
        plan = planner.plan_without_incremental_goal_memory(self.user_task)
        logging.info("----------------------------Results:----------------------------------")
        logging.info("Task: " + str(self.user_task))
        logging.info("Generated Plan:\n" + str(plan))

        # find results of the plan_with_incremental_goal_memory
        prolog = PrologWrapper()
        generated_goal = planner.problem.goal
        logging.info("Generated Goal: " + str(generated_goal))
        logging.info("Desired Goal: " + str(self.goal_state))
        resulting_state = prolog.get_result_from_plan(self.scene.objects, self.scene.relations, plan)
        resulting_state = define_predicates(resulting_state)

        # evaluate_with_memory resulting state and goal
        if "free" in str(self.goal_state):
            success = "yes" == input("Manual cofirmation:\n" + str(plan))
        else:
            success = self._matches_goal(resulting_state)
        logging.info("Goal Reached: " + str(success))
        loops = planner.feedback_loops
        acc, prec, f1 = self._get_f1(generated_goal)
        return success, loops, acc, prec, f1

    def evaluate_incremental_goal_memory_with_memory(self, planner: AutoGPTPlanner) -> Tuple[bool, int, float, float, float, bool, float]:
        """
        Uses AutoGPTPlanner to generate a plan_with_incremental_goal_memory for the given user_task of this scenario and calculates different metrics
        :param planner: the planner that use
        :return: a tuple of whether the goal is sufficient, how many loops the planner took,
        the precision, the recall and the f1 score of the generated goal with the desired goal
        """
        logging.info("NEW CASE ------------------------------------------------------------------")
        # print(self.user_task)
        # generate the plan_with_incremental_goal_memory
        planner.reset_history()
        planner.update_scene(self.scene.objects, self.scene.relations, self.scene.get_locations())
        plan = planner.plan_with_incremental_goal_memory(self.user_task, 3)
        logging.info("----------------------------Results:----------------------------------")
        memory = IncrementalGoalMemory('../incremental_goal_memory/known_pairs.csv')
        logging.info("Task: " + str(self.user_task))
        logging.info("Generated Plan:\n" + str(plan))

        # find results of the plan_with_incremental_goal_memory
        prolog = PrologWrapper()
        generated_goal = planner.problem.goal
        logging.info("Generated Goal: " + str(generated_goal))
        logging.info("Desired Goal: " + str(self.goal_state))
        resulting_state = prolog.get_result_from_plan(self.scene.objects, self.scene.relations, plan)
        resulting_state = define_predicates(resulting_state)

        # evaluate resulting state and goal
        if "free" in str(self.goal_state):
            success = "yes" == input("Manual cofirmation:\n" + str(plan))
        else:
            success = self._matches_goal(resulting_state)
        logging.info("Goal Reached: " + str(success))
        loops = planner.feedback_loops
        acc, prec, f1 = self._get_f1(generated_goal)
        return success, loops, acc, prec, f1

    def validate(self, oam_db, asm):
        domain = define_domain("robotic_planning", oam_db, self.scene.objects, asm, False)
        problem = define_problem_goal("test", domain, self.scene.objects, self.scene.relations,
                                      self.scene.get_locations(), asm, goal=self.goal_state)
        self.feedback_loops = 0
        planner = FastDownwardPlanner(domain)
        plan = planner.solve(problem)
        print(plan)

    def _matches_goal(self, state: List[Predicate]):
        dnf = goal_to_dnf(self.goal_state)
        print("DNF = " + dnf_to_string(dnf))

        for sub_state in dnf:
            positive, negative = self._get_pos_neg(sub_state)
            for predicate in negative:
                if predicate in state:
                    return False
            if set(positive).issubset(set(state)):
                return True
        return False

    def _get_pos_neg(self, substate: List[LogicElement]):
        positive = [s for s in substate if isinstance(s, Predicate)]
        negative = [s.logic_elements[0] for s in substate if isinstance(s, Not)]
        return positive, negative

    def _matching_rate(self, state: List[Predicate]):
        dnf = goal_to_dnf(self.goal_state)
        return max([matching_rate(sub_state, state) for sub_state in dnf])

    def _get_f1(self, generated_goal: LogicElement):
        if generated_goal is None:
            return 0.0, 0.0, 0.0
        dnf = goal_to_dnf(self.goal_state)
        dnf_generated = goal_to_dnf(generated_goal)
        recall_matrix = [[matching_rate(a, b) for a in dnf_generated] for b in dnf]
        precision_matrix = [[matching_rate(b, a) for a in dnf_generated] for b in dnf]
        print(recall_matrix)
        print(precision_matrix)
        recall = float(np.mean([max(column) for column in zip(*recall_matrix)]))
        precision = float(np.mean([max(column) for column in zip(*precision_matrix)]))
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        return precision, recall, f1


def jaccard_similarity(set_a: set, set_b: set):
    """
    Retruns the jaccard similarity between two sets |A n B| / |A U B|
    :param set_a: the first set
    :param set_b: the second set
    :return: the jaccard similairty between 0 and 1
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
