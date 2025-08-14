# this file is used to evaluate the inner feedback loop for planning with autogpt_p
# evaluation criteria include: success rate, amount of loops needed for success,
# percentage of relevant predicates found, f1 score of predicates
from __future__ import annotations
import csv
import logging
from typing import List

from planning_memory.dynamic_actor_provider import DynamicActorProvider
from planning_memory.static_capability_provider import StaticCapabilityProvider

from autogpt_p.evaluation.autogpt_p_evaluation_config import AutoGPTPEvaluationConfig, ScenarioEnum, ModelEnum, \
    generate_all_combinations
from autogpt_p.evaluation.autogpt_p_evaluation_scenario import AutoGPTPEvaluationScenario

from autogpt_p.helpers.paths import *
from autogpt_p.input.simulated_user import SimulatedUser
from autogpt_p.llm.llm_factory import LLMFactory

from pddl.core import LogicOp, And, LogicElement, Predicate, Not
# from planning_memory.planning_memory_segment import ActorReader

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.execution.actor_skill_mapping import ActorSkillMapping
from autogpt_p.execution.pddl_scenario import define_domain, define_problem
from autogpt_p.helpers.scene_read_write import read_scene
from pddl.problem import parse_logic_element
import pandas as pd

from autogpt_p.simulation.autogpt_p_simulator import AutoGPTPSimulator
from autogpt_p.state_machine.auto_gpt_p_memory import Memory

FILE_FORMAT = "autogptp_{}_{}"


def collapse_goal(goal: LogicElement):
    if isinstance(goal, LogicOp):
        if len(goal.logic_elements) == 1 and not isinstance(goal, Not):
            return collapse_goal(goal.logic_elements[0])
        else:
            for i in range(len(goal.logic_elements)):
                goal.logic_elements[i] = collapse_goal(goal.logic_elements[i])
    return goal


def correct_goal(goal: LogicElement):
    if isinstance(goal, Predicate) or isinstance(goal, Not):
        return And([goal])
    else:
        return goal


class AutoGPTPEvaluation:
    """
    Class for evaluating the outer planing loop
    """

    def __init__(self, simulator: AutoGPTPSimulator, scenario_file: str):
        """
        Creates a new evaluator for a given planner
        :param simulator:
        :param scenario_file:
        """
        self.simulator = simulator
        self.scenarios = self._read_scenarios(scenario_file)

    def _read_scenarios(self, scenario_file) -> List[AutoGPTPEvaluationScenario]:
        with open(scenario_file, 'r') as file:
            reader = csv.DictReader(file)
            scenarios = []
            scenario_dir = os.path.dirname(scenario_file)

            for row in reader:
                # task and context values
                task = row['task']
                scene_file = row['scene_file']
                explored_locations_string = row['explored']
                explored_locations = explored_locations_string.split(" ")
                substitutions_file = row['substitutions_file']

                # load stuff from files
                relative_path_to_scene_dir = "../scenes"
                relative_path_to_substitutions_dir = "../allowed_substitutions"
                scenes_dir = os.path.join(scenario_dir, relative_path_to_scene_dir)
                substitutions_dir = os.path.join(scenario_dir, relative_path_to_substitutions_dir)
                objects, relations, locations = read_scene(os.path.join(scenes_dir, scene_file))
                simulated_user = SimulatedUser.from_file(os.path.join(substitutions_dir, substitutions_file))

                # goal parsing
                desired_goal = row['desired_goal']
                domain = define_domain("Evaluation", self.simulator.memory.oam_db, objects, generic_predicates=False)
                problem = define_problem("Test", domain, objects, relations, locations,
                                         self.simulator.memory.actor_skill_mapping, [],
                                         "")
                goal = parse_logic_element(desired_goal, domain.predicates, problem.objects)
                goal = correct_goal(collapse_goal(goal))

                print("Collapsed Goal " + str(goal))
                min_tool_calls = row['min_tool_calls']
                min_costs = row['min_costs']
                print(scene_file)
                scenarios.append(
                    AutoGPTPEvaluationScenario(SimulatedScene(objects, relations, locations), explored_locations,
                                               simulated_user.allowed_substitutions, task, goal, min_tool_calls,
                                               min_costs))

            return scenarios

    def evaluate(self):
        """

        :return:
        """
        evaluation_data = [scenario.evaluate(self.simulator) for scenario in self.scenarios]
        print(evaluation_data)

        df = pd.DataFrame.from_records(evaluation_data,
                                       columns=['success', 'plan_is_min', 'plan_cost_rate',
                                                'tools_is_min', 'tools_cost_rate'])
        success_rate = df['success'].mean()
        plan_is_min_rate = df['plan_is_min'].mean()
        plan_average_cost_rate_success = df.loc[df['success'], 'plan_cost_rate'].mean()
        tools_is_min_rate = df['tools_is_min'].mean()
        tools_average_cost_rate_success = df.loc[df['success'], 'tools_cost_rate'].mean()
        return success_rate, plan_is_min_rate, plan_average_cost_rate_success, \
            tools_is_min_rate, tools_average_cost_rate_success

    def evaluate_n(self, times):
        """

        :param times:
        :return:
        """
        evaluation_data = [self.evaluate() for _ in range(times)]
        df = pd.DataFrame.from_records(evaluation_data,
                                       columns=['success', 'plan_is_min', 'plan_cost_rate',
                                                'tools_is_min', 'tools_cost_rate'])
        mean_values = df.mean()
        return tuple(mean_values.iloc[0])

    def validate(self):
        [scenario.validate(self.simulator.memory.oam_db, self.simulator.memory.actor_skill_mapping) for scenario in self.scenarios]

    @classmethod
    def from_config(cls, config: AutoGPTPEvaluationConfig) -> AutoGPTPEvaluation:
        if config.model == ModelEnum.GPT_4:
            model = "4"
        elif config.model == ModelEnum.GPT_5:
            model = "5"
        else:
            model = "3"

        if LLMFactory.get_instance() is None:
            LLMFactory("GPT", model)
        else:
            LLMFactory.get_instance().version = model

        if config.scenario == ScenarioEnum.ALTERNATIVE:
            scenario = "autogptp_alternative.csv"
        elif config.scenario == ScenarioEnum.EXPLORATION:
            scenario = "autogptp_exploration.csv"
        elif config.scenario == ScenarioEnum.COMPLEX:
            scenario = "autogptp_complex.csv"
        elif config.scenario == ScenarioEnum.PLAN:
            scenario = "autogptp_plan.csv"
        elif config.scenario == ScenarioEnum.PARTIAL:
            scenario = "autogptp_partial.csv"
        else:
            scenario = "test.csv"

        base_path = SCENARIO_DIR

        capabilities = StaticCapabilityProvider("../../../planning_memory/data/capabilities/all_capabilities.json")
        capabilities.process_skills()
        actor_provider = DynamicActorProvider("robot0", "robot_profile", "robot", capabilities)
        actor_skill_mapping = ActorSkillMapping([actor_provider.get_actor()])

        memory = Memory.empty_memory()
        memory.actor_skill_mapping = actor_skill_mapping

        simulator = AutoGPTPSimulator(memory, False)

        print("crated sim")

        return AutoGPTPEvaluation(simulator, os.path.join(base_path, scenario))


def get_filename_without_extension(path):
    base_name = os.path.basename(path)  # Gets filename with extension
    file_name_without_extension = os.path.splitext(base_name)[0]  # Splits filename from its extension
    return file_name_without_extension


def write_results(filename, results):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write a header (optional)
        writer.writerow(['success', 'plan_is_min', 'plan_cost_rate', 'tools_is_min', 'tools_cost_rate'])
        # Write each tuple as a row in the CSV
        writer.writerow(results)


if __name__ == "__main__":
    config = AutoGPTPEvaluationConfig(ModelEnum.GPT_5, ScenarioEnum.COMPLEX)
    file_name = FILE_FORMAT.format(config.model, config.scenario)
    logging.basicConfig(filename=os.path.join(AUTOGPTP_LOGS_DIR, file_name + ".log"),
                        level=logging.INFO,
                        format='%(message)s')
    results_file_path = os.path.join(AUTOGPTP_RESULTS_DIR, file_name + ".csv")
    evaluation = AutoGPTPEvaluation.from_config(config)
    result = evaluation.evaluate()
    write_results(results_file_path, result)
    # evaluation.validate()
