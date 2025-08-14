from __future__ import annotations
import csv
import logging
import os
from typing import List

import pandas as pd
from autogpt_p.evaluation.alternative_suggestion_evaluation_config import AlternativeSuggestionEvaluationConfig, \
    ModelEnum, MethodEnum, ScenarioEnum
from autogpt_p.substitution.affordance_based_substituiton import AffordanceBasedSubstitution
from autogpt_p.substitution.naive_substitution import NaiveSubstitution
from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceMappingDatabase

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.evaluation.alternative_suggestion_scenario import AlternativeSuggestionScenario
from autogpt_p.helpers.paths import SCENARIO_DIR, ALTERNATIVE_SUGGESTION_LOGS_DIR, \
    ALTERNATIVE_SUGGESTION_RESULTS_DIR
from autogpt_p.helpers.scene_read_write import read_scene
from autogpt_p.llm.chat_gpt_interface import ChatGPTInterface, GPT_4, GPT_3
from autogpt_p.substitution.substitution import Substitution
from autogpt_p.substitution.substitution_memory import SubstitutionMemory

FILE_FORMAT = "alternative_suggestion_{}_{}_{}"


def _read_scenarios(scenario_file) -> List[AlternativeSuggestionScenario]:
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
            missing_object = row['missing_object']
            alternatives = row['alternatives']
            scenarios.append(
                AlternativeSuggestionScenario(SimulatedScene(objects, relations, locations), task, missing_object,
                                              parse_alternatives(alternatives)))
        return scenarios


def parse_alternatives(alternatives: str) -> List[str]:
    return alternatives.split(" ")


def get_filename_without_extension(file_path: str):
    base_name = os.path.basename(file_path)  # Gets filename with extension
    file_name_without_extension = os.path.splitext(base_name)[0]  # Splits filename from its extension
    return file_name_without_extension


def write_results(filename, results):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write a header (optional)
        writer.writerow(['success_rate'])
        # Write each tuple as a row in the CSV
        writer.writerow(results)


class AlternativeSuggestionEvaluation:
    """
    Class for evaluating the inner planing loop
    """

    def __init__(self, substitution: Substitution, scenario_file: str):
        """
        Creates a new evaluator for a given planner
        :param substitution: the planner to be evaluated
        :param scenario_file:
        """
        self.substitution = substitution
        self.scenarios = _read_scenarios(scenario_file)

    def evaluate(self) -> float:
        """

        :return:
        """
        evaluation_data = [scenario.evaluate_substitution(self.substitution) for scenario in self.scenarios]
        return sum(evaluation_data) / len(evaluation_data)

    def evaluate_n(self, times) -> float:
        """

        :param times:
        :return:
        """
        evaluation_data = [self.evaluate() for _ in range(times)]
        df = pd.DataFrame.from_records(evaluation_data, columns=['success_rate'])
        mean_values = df.mean()
        return mean_values.iloc[0]

    @classmethod
    def from_config(cls, config: AlternativeSuggestionEvaluationConfig) -> AlternativeSuggestionEvaluation:
        if config.model == ModelEnum.GPT_4:
            model = GPT_4
        else:
            model = GPT_3

        if config.method == MethodEnum.NAIVE:
            substitution = NaiveSubstitution(ChatGPTInterface(model=model), SubstitutionMemory(), False)
        else:
            oam_db = ObjectAffordanceMappingDatabase.load_from_data("simulation_classes_.json",
                                                                    "proposed_affordances_alternative.json",
                                                                    "simulation_classes_ground_truth_.json")
            substitution = AffordanceBasedSubstitution(ChatGPTInterface(model=model), SubstitutionMemory(), oam_db,
                                                       False)

        if config.scenario == ScenarioEnum.SIMPLE:
            scenario = "substitution_simple.csv"
        elif config.scenario == ScenarioEnum.MEDIUM:
            scenario = "substitution_medium.csv"
        elif config.scenario == ScenarioEnum.COMPLEX:
            scenario = "substitution_complex.csv"
        else:
            scenario = "test.csv"

        return AlternativeSuggestionEvaluation(substitution, os.path.join(SCENARIO_DIR, scenario))


if __name__ == "__main__":
    config = AlternativeSuggestionEvaluationConfig(ModelEnum.GPT_4, MethodEnum.AFFORDANCE, ScenarioEnum.COMPLEX)

    file_name = FILE_FORMAT.format(config.model, config.method, config.scenario)
    logging.basicConfig(filename=os.path.join(ALTERNATIVE_SUGGESTION_LOGS_DIR, file_name + ".log"),
                        level=logging.INFO,
                        format='%(message)s')
    results_file_path = os.path.join(ALTERNATIVE_SUGGESTION_RESULTS_DIR, file_name + ".csv")

    evaluation = AlternativeSuggestionEvaluation.from_config(config)
    result = evaluation.evaluate()
    write_results(results_file_path, [result])
