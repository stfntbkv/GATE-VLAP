import csv
import os

from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceDatabase

from chat_gpt_oam.chat_gpt_object_affordance_mapping import ChatGPTOAM, list_objects_mode, list_affordances_mode, \
    combination_mode
from tools.evalaution_config import EvaluationConfig, ModelEnum, AffordancesEnum, MethodEnum
from tools.json_csv import convert_results

OAM_VAR = "OAM_ROOT"
OAM_PACKAGE = os.environ[OAM_VAR]
CHAT_GPT_OAM_PACKAGE = os.path.join(OAM_PACKAGE, "python/chat_gpt_oam")
DATA_PATH = os.path.join(CHAT_GPT_OAM_PACKAGE, "data")
AFFORDANCES_PATH = os.path.join(DATA_PATH, "affordances")
CLASSES_PATH = os.path.join(DATA_PATH, "classes")
LOGS_PATH = os.path.join(CHAT_GPT_OAM_PACKAGE, "logs")
EVALUATION_PATH = os.path.join(CHAT_GPT_OAM_PACKAGE, "evaluation")
GT_PATH = os.path.join(CHAT_GPT_OAM_PACKAGE, "ground_truth")
RESULTS_PATH = os.path.join(CHAT_GPT_OAM_PACKAGE, "results/evaluation")

SIMULATION_GT = "simulation_classes_ground_truth.csv"
SIMULATION_GT_BASIC = "simulation_classes_ground_truth_basic.csv"

BASIC_AFFORDANCES = "basic_affordances"
PROPOSED_AFFORDANCES = "proposed_affordances"
COMBINATION_SUFFIX = "_alternative"

EXPERIMENT_FORMAT = "oam_simulation_{}_{}_{}"
LOG_EXT = ".log"
JSON_EXT = ".json"
CSV_EXT = ".csv"


def calculate_metrics(ground_truth_file, generated_file):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    stats = {}
    stats2 = {}

    with open(ground_truth_file, 'r') as gt_file:
        with open(generated_file, 'r') as gen_file:
            gt_reader = csv.reader(gt_file)
            gen_reader = csv.reader(gen_file)

            for gt_row in gt_reader:
                if gt_row in gen_reader:
                    true_positives += 1
                else:
                    false_negatives += 1
                    a = gt_row[1]
                    stats2[a] = stats2[a] + 1 if a in stats2 else 1
                    if gt_row[1] != "X":
                        print(str(gt_row) + " was not detected")
                gen_file.seek(0)  # reset the file pointer to the beginning
            gt_file.seek(0)

            for gen_row in gen_reader:
                if gen_row not in gt_reader:
                    if gen_row[1] != "X":
                        print(str(gen_row) + " was wrongly detected")
                    a = gen_row[1]
                    stats[a] = stats[a] + 1 if a in stats else 1
                    false_positives += 1
                gt_file.seek(0)

    print(true_positives)
    print(false_negatives)
    print(false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    sorted_dict = dict(sorted(stats.items(), key=lambda x: x[1]))
    print(sorted_dict)
    sorted_dict = dict(sorted(stats2.items(), key=lambda x: x[1]))
    print(sorted_dict)

    return precision, recall, f1_score


class OAMEvaluation:

    def __init__(self, ground_truth_file: str, chat_gpt_oam: ChatGPTOAM, method: MethodEnum, oam_file: str,
                 result_file: str):
        self.ground_truth_file = ground_truth_file
        self.oam = chat_gpt_oam
        self.method = method
        self.oam_file = oam_file
        self.result_file = result_file

    def _run_oam(self):
        if self.method == MethodEnum.LIST_OBJECTS:
            self.oam.generate_oams(mode=list_objects_mode)
        elif self.method == MethodEnum.LIST_AFFORDANCES:
            self.oam.generate_oams(mode=list_affordances_mode)
        elif self.method == MethodEnum.BINARY:
            self.oam.generate_oams(mode=combination_mode)
        else:
            self.oam.generate_oams(mode=combination_mode)
        self.oam.write_json(self.oam_file)
        convert_results(self.oam_file.replace(".json", ""))

    def evaluate(self):
        if not self._result_exists():
            self._run_oam()
        metrics = calculate_metrics(self.ground_truth_file, self.oam_file.replace(".json", ".csv"))
        precision, recall, f1_score = metrics
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1_score:.2f}')
        self._write_to_csv(metrics)

    def _result_exists(self) -> bool:
        return os.path.exists(self.oam_file.replace(".json", ".csv"))

    def _write_to_csv(self, metrics):
        with open(self.result_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write a header (optional)
            writer.writerow(['precision', 'recall', 'f1_score'])
            # Write each tuple as a row in the CSV
            writer.writerow(metrics)

    @classmethod
    def from_config(cls, config: EvaluationConfig):
        model = "gpt-4" if config.model == ModelEnum.GPT_4 else "gpt-3.5-turbo-0613"
        gt_file = SIMULATION_GT_BASIC if config.affordances == AffordancesEnum.BASIC else SIMULATION_GT
        gt_file = os.path.join(GT_PATH, gt_file)
        classes_file = os.path.join(CLASSES_PATH, "simulation_classes.json")
        affordances_file = BASIC_AFFORDANCES if config.affordances == AffordancesEnum.BASIC else PROPOSED_AFFORDANCES
        affordances_file += COMBINATION_SUFFIX if config.method == MethodEnum.BINARY_LOGIC else ""
        affordances_file += JSON_EXT
        affordances_file = os.path.join(AFFORDANCES_PATH, affordances_file)
        oa_db = ObjectAffordanceDatabase(classes_file, affordances_file)
        file_name = EXPERIMENT_FORMAT.format(str(config.model), str(config.method), str(config.affordances))
        oam_file_name = os.path.join(RESULTS_PATH, file_name + JSON_EXT)
        log_file_name = os.path.join(LOGS_PATH, file_name + LOG_EXT)
        results_file_name = os.path.join(EVALUATION_PATH, file_name + CSV_EXT)
        oam = ChatGPTOAM(oa_db, model, log_file_name)
        return OAMEvaluation(gt_file, oam, config.method, oam_file_name, results_file_name)


def main():
    config = EvaluationConfig(ModelEnum.GPT_4, MethodEnum.BINARY, AffordancesEnum.BASIC)
    evaluation = OAMEvaluation.from_config(config)
    evaluation.evaluate()


if __name__ == '__main__':
    main()
