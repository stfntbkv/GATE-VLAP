import logging
from typing import Tuple, List

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.execution.actor_skill_mapping import ActorSkillMapping
from autogpt_p.execution.pddl_scenario import define_predicates, define_domain, define_problem_goal

from autogpt_p.helpers.dnf import goal_to_dnf
from autogpt_p.input.input_manager_factory import InputManagerFactory
from autogpt_p.input.simulated_user import SimulatedUser
from pddl.core import LogicElement, Predicate, Not
from autogpt_p.planning.planner import FastDownwardPlanner
from autogpt_p.simulation.autogpt_p_simulator import AutoGPTPSimulator


def dnf_to_string(dnf: List[List[Predicate]]) -> str:
    return str([[str(pred) for pred in substate] for substate in dnf])


class AutoGPTPEvaluationScenario:
    """

    """

    def __init__(self, scene: SimulatedScene, explored_locations: List[str],
                 allowed_substitutions: List[Tuple[str, str]],
                 user_task: str, goal_state: LogicElement, min_tool_calls: int, min_costs: int):
        """
        Creates a new evaluation scenario
        :param user_task: the natural language description of what the user wants
        :param goal_state: the ideal goal state set by the evaluation designer of what the user wants with his task
        """
        self.scene = scene
        self.explored_locations = explored_locations
        self.allowed_substitutions = allowed_substitutions
        self.user_task = user_task
        self.goal_state = goal_state
        self.min_tool_calls = min_tool_calls
        self.min_costs = min_costs

    def evaluate(self, simulator: AutoGPTPSimulator) -> Tuple[bool, bool, float, bool, float]:
        """
        Uses AutoGPTPlanner to generate a plan for the given user_task of this scenario and calculates different metrics
        :param simulator:
        :return: a tuple of whether the goal is sufficient, how many loops the planner took,
        the precision, the recall and the f1 score of the generated goal with the desired goal
        """
        logging.info("--------------------------------------NEW CASE ----------------------------------------")
        print(self.user_task)
        # reset simulator to be sure
        simulator.reset_memory()
        simulator.reset_state_machine()

        # prepare simulator for command
        simulator.set_scene(self.scene)

        if self.explored_locations == ["all"]:
            simulator.simulate_full_exploration()
        else:
            simulator.simulate_exploration(self.explored_locations)

        if InputManagerFactory.get_instance() is None:
            InputManagerFactory("")
        InputManagerFactory.get_instance().set_dummy(SimulatedUser(self.allowed_substitutions))

        # execute command
        simulator.command(self.user_task)

        logging.info("---------------------------------------Results:---------------------------------------------")
        # check results
        resulting_state = simulator.scene.relations
        resulting_state = define_predicates(resulting_state)

        # evaluate resulting state and goal
        plan = simulator.get_last_executed_plan()
        if "free" in str(self.goal_state):
            success = "yes" == input("Manual cofirmation:\n" + str(plan))
        else:
            success = self._matches_goal(resulting_state)

        logging.info("Task: " + str(self.user_task))
        logging.info("Generated Plan:\n" + str(plan))
        logging.info("Desired Goal: " + str(self.goal_state))
        logging.info("Goal Reached: " + str(success))

        plan_length = plan.get_real_length()
        is_min = plan_length <= int(self.min_costs)
        logging.info("Plan costs are: {} -- minimal costs: {}".format(plan_length, self.min_costs))
        logging.info("Plan Costs minimal: " + str(is_min))
        plan_cost_rate = max(plan_length / float(self.min_costs), 1.0) if float(self.min_costs) != 0.0 else \
            1.0 if plan_length == 0 else 0
        logging.info("Plan Costs Rate: " + str(plan_cost_rate))

        tools_used = len(simulator.memory.tool_memory)
        min_tools = (tools_used <= int(self.min_tool_calls))
        tool_cost_rate = max(tools_used / float(self.min_tool_calls), 1.0)
        logging.info("Used tools " + ", ".join([t.keyword for t in simulator.memory.tool_memory]))
        logging.info("Tool Costs are: {} -- minimal costs: {}".format(tools_used, self.min_tool_calls))
        logging.info("Tool Costs minimal: " + str(min_tools))
        logging.info("Costs Rate: " + str(tool_cost_rate))
        logging.info("----------------------------------------------------------------------------")
        return success, is_min, plan_cost_rate, min_tools, tool_cost_rate

    def validate(self, oam_db, asm: ActorSkillMapping):
        domain = define_domain("robotic_planning", oam_db, self.scene.objects, asm, False)
        problem = define_problem_goal("test", domain, self.scene.objects, self.scene.relations,
                                      self.scene.get_locations(), asm, goal=self.goal_state)
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
