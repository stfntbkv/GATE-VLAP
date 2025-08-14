# this file is used to evaluate_with_memory the inner feedback loop for planning with autogpt_p
# evaluation criteria include: success rate, amount of loops needed for success,
# percentage of relevant predicates found, f1 score of predicates
from __future__ import annotations
import csv
import logging
from typing import List

from autogpt_p.evaluation.incremental_goal_evaluation_config import IncrementalGoalMemoryEvaluationConfig, ModelEnum, AutoregressionEnum, \
    ScenarioEnum, ClassesEnum
from autogpt_p.helpers.paths import *

from pddl.core import LogicOp, And, LogicElement, Predicate, Not
from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceMappingDatabase
from planning_memory.dynamic_actor_provider import DynamicActorProvider
from planning_memory.static_capability_provider import StaticCapabilityProvider, CAPABILITIES_FILE

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.execution.actor_skill_mapping import ActorSkillMapping
from autogpt_p.llm.chat_gpt_interface import ChatGPTInterface, GPT_4, GPT_3
from autogpt_p.execution.pddl_scenario import define_domain, define_problem
from autogpt_p.planning.autogpt_planner import AutoGPTPlanner
from autogpt_p.evaluation.incremental_goal_evaluation_scenario import IncrementalGoalMemoryEvaluationScenario
from autogpt_p.helpers.scene_read_write import read_scene
from pddl.problem import parse_logic_element
import pandas as pd

FILE_FORMAT = "planning_{}_{}_{}_{}"


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


class IncrementalGoalMemoryEvaluation:
    """
    Class for evaluating the use of the incremental goal memory
    """

    def __init__(self, auto_gpt_planner: AutoGPTPlanner, actor_skill_mapping: ActorSkillMapping, scenario_file: str):
        """
        Creates a new evaluator for a given planner.
        :param auto_gpt_planner: the planner to be evaluated
        :param scenario_file:
        """
        self.planner = auto_gpt_planner
        self.actor_skill_mapping = actor_skill_mapping
        self.scenarios = self._read_scenarios(scenario_file)

    def _read_scenarios(self, scenario_file) -> List[IncrementalGoalMemoryEvaluationScenario]:
        with open(scenario_file, 'r') as file:
            reader = csv.DictReader(file)
            scenarios = []
            # skip the header line
            scenario_dir = os.path.dirname(scenario_file)

            for row in reader:
                task = row['task']
                scene_file = row['scene_file']
                relative_path_to_scene_dir = "../scenes"
                scenes_dir = os.path.join(scenario_dir, relative_path_to_scene_dir)
                objects, relations, locations = read_scene(os.path.join(scenes_dir, scene_file))
                domain = define_domain("Evaluation", self.planner.oam_db, objects)
                problem = define_problem("Test", domain, objects, relations, locations, self.actor_skill_mapping, [],
                                         "")
                desired_goal = row['desired_goal']
                goal = parse_logic_element(desired_goal, domain.predicates, problem.objects)
                print(str(goal))
                goal = correct_goal(collapse_goal(goal))
                print("Collapsed Goal " + str(goal))
                scenarios.append(
                    IncrementalGoalMemoryEvaluationScenario(SimulatedScene(objects, relations, locations), task, goal))

            return scenarios

    def evaluate_with_memory(self):
        """

        :return:
        """
        evaluation_data = [scenario.evaluate_incremental_goal_memory_with_memory(self.planner) for scenario in self.scenarios]
        # print(evaluation_data)

        df = pd.DataFrame.from_records(evaluation_data,
                                       columns=['success', 'loops', 'acc', 'prec', 'f1'])
        success_rate = df['success'].mean()
        average_loops = df['loops'].mean()
        average_loops_success = df.loc[df['success'], 'loops'].mean()
        average_acc = df['acc'].mean()
        average_prec = df['prec'].mean()
        average_f1 = df['f1'].mean()
        return success_rate, average_loops, average_loops_success, average_acc, average_prec, average_f1


    def evaluate_without_memory(self):
        """

        :return:
        """
        evaluation_data = [scenario.evaluate_incremental_goal_memory_without_memory(self.planner) for scenario in self.scenarios]
        # print(evaluation_data)

        df = pd.DataFrame.from_records(evaluation_data,
                                       columns=['success', 'loops', 'acc', 'prec', 'f1'])
        success_rate = df['success'].mean()
        average_loops = df['loops'].mean()
        average_loops_success = df.loc[df['success'], 'loops'].mean()
        average_acc = df['acc'].mean()
        average_prec = df['prec'].mean()
        average_f1 = df['f1'].mean()
        return success_rate, average_loops, average_loops_success, average_acc, average_prec, average_f1

    def validate(self):
        [scenario.validate(self.planner.oam_db, self.actor_skill_mapping) for scenario in self.scenarios]

    def evaluate_n(self, times):
        """

        :param times:
        :return:
        """
        # TODO fehlt nochmal diese Methode aber ohne memory? Andererseits wird diese Methode nie verwendet?
        evaluation_data = [self.evaluate_with_memory() for _ in range(times)]
        df = pd.DataFrame.from_records(evaluation_data, columns=['success_rate', 'average_loops',
                                                                 'average_loops_success', 'average_acc', 'average_prec',
                                                                 'average_f1'])
        mean_values = df.mean()
        return tuple(mean_values.iloc[0])

    @classmethod
    def from_config(cls, config: IncrementalGoalMemoryEvaluationConfig) -> IncrementalGoalMemoryEvaluation:
        if config.model == ModelEnum.GPT_4:
            model = GPT_4
        else:
            model = GPT_3

        autoregression = config.autoregressive == AutoregressionEnum.ON

        if config.classes == ClassesEnum.SAYCAN:
            classes_file = "saycan_classes.json"
            oam_file = "saycan_oam.json"
        else:
            classes_file = "simulation_classes_.json"
            oam_file = "simulation_classes_2.json"

        if config.scenario == ScenarioEnum.SAYCAN_NL_NOUN:
            scenario = "saycan_c_nl_noun.csv"
        elif config.scenario == ScenarioEnum.SAYCAN_NL_PRIMITIVE:
            scenario = "saycan_a_nl_primitive.csv"
        elif config.scenario == ScenarioEnum.SAYCAN_NL_VERB:
            scenario = "saycan_b_nl_verb.csv"
        elif config.scenario == ScenarioEnum.SAYCAN_STRUCTURED_LANGUAGE:
            scenario = "saycan_d_structured_language.csv"
        elif config.scenario == ScenarioEnum.SAYCAN_EMBODIMENT:
            scenario = "saycan_e_embodiment.csv"
        elif config.scenario == ScenarioEnum.SAYCAN_CROWD_SOURCED:
            scenario = "saycan_f_crowd_sourced.csv"
        elif config.scenario == ScenarioEnum.SAYCAN_DRAWER:
            scenario = "saycan_x_drawer.csv"
        elif config.scenario == ScenarioEnum.SAYCAN_LONG_HORIZON:
            scenario = "saycan_g_long_horizon.csv"
        elif config.scenario == ScenarioEnum.SIMPLE_GOAL:
            scenario = "evaluation_simple_goal.csv"
        elif config.scenario == ScenarioEnum.SIMPLE_TASK:
            scenario = "evaluation_simple_task.csv"
        elif config.scenario == ScenarioEnum.COMPLEX_GOAL:
            scenario = "evaluation_complex_goal.csv"
        elif config.scenario == ScenarioEnum.KNOWLEDGE:
            scenario = "evaluation_knowledge.csv"
        elif config.scenario == ScenarioEnum.IMPLICIT:
            scenario = "evaluation_implicit.csv"
        elif config.scenario == ScenarioEnum.MEMORY:
            scenario = "evaluation_memory.csv"
        elif config.scenario == ScenarioEnum.MEMORY_OTHER_ENVIRONMENT:
            scenario = "evaluation_memory_other_environment.csv"
        elif config.scenario == ScenarioEnum.MEMORY_OTHER_OBJECTS:
            scenario = "evaluation_memory_other_objects.csv"
        elif config.scenario == ScenarioEnum.MEMORY_REFORMATED:
            scenario = "evaluation_memory_reformated.csv"
        else:
            scenario = "test.csv"

        base_path = os.path.join(SAYCAN_DIR, "scenarios") if str(config.scenario).startswith("saycan") else SCENARIO_DIR

        capabilities = StaticCapabilityProvider("../../../planning_memory/data/capabilities/all_capabilities.json")
        capabilities.process_skills()
        actor_provider = DynamicActorProvider("robot0", "robot_profile", "robot", capabilities)
        actor_skill_mapping = ActorSkillMapping([actor_provider.get_actor()])
        llm = ChatGPTInterface(model)
        oamdb = ObjectAffordanceMappingDatabase.load_from_data(classes_file,
                                                               "proposed_affordances_alternative.json",
                                                               oam_file)
        if autoregression:
            max_loops = 10
        else:
            max_loops = 1

        planner = AutoGPTPlanner(llm, [], [], [], oamdb, actor_skill_mapping=actor_skill_mapping, max_loops=max_loops)

        return IncrementalGoalMemoryEvaluation(planner, actor_skill_mapping, os.path.join(base_path, scenario))


def get_filename_without_extension(path):
    base_name = os.path.basename(path)  # Gets filename with extension
    file_name_without_extension = os.path.splitext(base_name)[0]  # Splits filename from its extension
    return file_name_without_extension


def write_both_results(filename, results_without_memory, results_with_memory):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write a header (optional)
        writer.writerow(['success_rate', 'average_loops', 'average_loops_success', 'average_acc', 'average_prec',
                         'average_f1'])
        # Write each tuple as a row in the CSV
        writer.writerow(results_without_memory)
        writer.writerow(results_with_memory)
def write_with_memory_results(filename, results_with_memory):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write a header (optional)
        writer.writerow(['success_rate', 'average_loops', 'average_loops_success', 'average_acc', 'average_prec',
                         'average_f1'])
        # Write each tuple as a row in the CSV
        writer.writerow(results_with_memory)
def write_without_memory_results(filename, results_without_memory):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write a header (optional)
        writer.writerow(['success_rate', 'average_loops', 'average_loops_success', 'average_acc', 'average_prec',
                         'average_f1'])
        # Write each tuple as a row in the CSV
        writer.writerow(results_without_memory)

if __name__ == "__main__":
    configs = [IncrementalGoalMemoryEvaluationConfig(ModelEnum.GPT_4, AutoregressionEnum.ON,
                                     ClassesEnum.SIMULATION, ScenarioEnum.MEMORY_REFORMATED)]
    for config in configs:
        file_name = FILE_FORMAT.format(config.model, config.autoregressive, config.classes, config.scenario)
        os.remove('../../logs/incremental_goal_memory/planning_gpt_4_on_simulation_memory_reformated.log')
        logging.basicConfig(filename=os.path.join(INCREMENTAL_GOAL_MEMORY_LOGS_DIR, file_name + ".log"),
                            level=logging.INFO,
                            format='%(message)s')
        results_file_path = os.path.join(INCREMENTAL_GOAL_MEMORY_RESULTS_DIR, file_name + ".csv")

        evaluation = IncrementalGoalMemoryEvaluation.from_config(config)
        # choose the evaluation option
        evaluate_both = False
        evaluate_only_with_memory = True
        evaluate_only_without_memory = False
        if evaluate_both==True:
            results_without_memory = evaluation.evaluate_without_memory()
            results_with_memory = evaluation.evaluate_with_memory()
            write_both_results(results_file_path, results_without_memory, results_with_memory)
        elif evaluate_only_with_memory==True:
            results_with_memory = evaluation.evaluate_with_memory()
            write_with_memory_results(results_file_path, results_with_memory)
        else:
            results_without_memory = evaluation.evaluate_without_memory()
            write_without_memory_results(results_file_path, results_without_memory)

    # evaluation.validate()
