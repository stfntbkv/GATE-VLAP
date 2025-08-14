import csv
import logging
import os
import sys
import argparse
from typing import List
import pandas as pd

from autogpt_p.evaluation.libero_planner_evaluation_config import (
    LiberoPlannerEvaluationConfig, ModelEnum, AutoregressionEnum, 
    ClassesEnum, LiberoSuiteEnum
)
from autogpt_p.evaluation.libero_planner_evaluation_scenario import (
    LiberoPlannerEvaluationScenario, LiberoSimulatedScene
)
from autogpt_p.evaluation.libero_bddl_converter import BDDLConverter
from autogpt_p.helpers.paths import *

from pddl.core import LogicOp, And, LogicElement, Predicate, Not
from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceMappingDatabase
from planning_memory.dynamic_actor_provider import DynamicActorProvider
from planning_memory.static_capability_provider import StaticCapabilityProvider

from autogpt_p.execution.actor_skill_mapping import ActorSkillMapping
from autogpt_p.llm.chat_gpt_interface import ChatGPTInterface, GPT_4, GPT_3, GPT_5
from autogpt_p.llm.gemini_interface import (
    GeminiInterface, GEMINI_1_5_FLASH, GEMINI_1_5_PRO, GEMINI_1_5_FLASH_8B,
    GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_EXP, GEMINI_2_5_FLASH, GEMINI_2_5_PRO,
    GEMINI_EXP_1121, GEMINI_EXP_1114
)
from autogpt_p.execution.pddl_scenario import define_domain, define_problem
from autogpt_p.planning.autogpt_planner import AutoGPTPlanner
from autogpt_p.helpers.scene_read_write import read_scene
from pddl.problem import parse_logic_element


LIBERO_FILE_FORMAT = "libero_planning_{}_{}_{}_{}"


def collapse_goal(goal: LogicElement):
    """Simplify nested logical structures by removing unnecessary wrapper operations"""
    if isinstance(goal, LogicOp):
        if len(goal.logic_elements) == 1 and not isinstance(goal, Not):
            return collapse_goal(goal.logic_elements[0])
        else:
            for i in range(len(goal.logic_elements)):
                goal.logic_elements[i] = collapse_goal(goal.logic_elements[i])
    return goal


def correct_goal(goal: LogicElement):
    """Ensure all goals are properly wrapped in an AND operation for consistent processing"""
    if isinstance(goal, Predicate) or isinstance(goal, Not):
        return And([goal])
    else:
        return goal


class LiberoPlannerEvaluation:
    """
    Class for evaluating the planner on LIBERO robotic manipulation tasks
    """

    def __init__(self, auto_gpt_planner: AutoGPTPlanner, actor_skill_mapping: ActorSkillMapping, 
                 scenario_file: str, libero_suite: str):
        """
        Creates a new LIBERO evaluator for a given planner
        :param auto_gpt_planner: the planner to be evaluated
        :param actor_skill_mapping: robot actor skill mapping
        :param scenario_file: path to CSV file containing LIBERO scenarios
        :param libero_suite: name of the LIBERO suite being evaluated
        """
        self.planner = auto_gpt_planner
        self.actor_skill_mapping = actor_skill_mapping
        self.libero_suite = libero_suite
        self.scenarios = self._read_libero_scenarios(scenario_file)

    def _read_libero_scenarios(self, scenario_file) -> List[LiberoPlannerEvaluationScenario]:
        """Read and parse LIBERO scenarios from CSV file"""
        with open(scenario_file, 'r') as file:
            reader = csv.DictReader(file)
            scenarios = []
            scenario_dir = os.path.dirname(scenario_file)

            for row in reader:
                task = row['task']
                scene_file = row['scene_file']
                
                # Construct path to scene files
                relative_path_to_scene_dir = "../scenes"
                scenes_dir = os.path.join(scenario_dir, relative_path_to_scene_dir)
                scene_path = os.path.join(scenes_dir, scene_file)
                
                try:
                    # Read scene file
                    objects, relations, locations = read_scene(scene_path)
                    
                    # Create LIBERO domain for parsing
                    domain = define_domain("LIBERO_Evaluation", self.planner.oam_db, objects)
                    problem = define_problem("LIBERO_Test", domain, objects, relations, locations, 
                                           self.actor_skill_mapping, [], "")
                    
                    # Parse and normalize goal
                    desired_goal = row['desired_goal']
                    goal = parse_logic_element(desired_goal, domain.predicates, problem.objects)
                    goal = correct_goal(collapse_goal(goal))
                    
                    # Debug output
                    goal_str = str(goal)
                    print(f"LIBERO Goal: {goal_str}")
                    
                    # Validate goal reconstruction
                    try:
                        goal2 = LogicOp.from_string(goal_str)
                        print(f"Reconstructed: {str(goal2)}")
                        print(f"Goals match: {str(goal2) == str(goal)}")
                    except Exception as e:
                        print(f"Goal reconstruction warning: {e}")
                    
                    min_costs = int(row['min_costs'])
                    
                    # Create LIBERO scenario
                    libero_scene = LiberoSimulatedScene(objects, relations, locations)
                    scenarios.append(
                        LiberoPlannerEvaluationScenario(libero_scene, task, goal, min_costs)
                    )
                    
                except Exception as e:
                    print(f"Error processing scenario {scene_file}: {e}")
                    continue

            print(f"Loaded {len(scenarios)} LIBERO scenarios from {self.libero_suite}")
            return scenarios

    def evaluate(self):
        """
        Run evaluation on all LIBERO scenarios and compute aggregate metrics
        :return: tuple of aggregate metrics
        """
        evaluation_data = [scenario.evaluate_planner(self.planner) for scenario in self.scenarios]
        print(f"LIBERO Evaluation Data: {evaluation_data}")

        df = pd.DataFrame.from_records(evaluation_data,
                                       columns=['success', 'loops', 'acc', 'prec', 'f1', 'is_min', 'cost_rate'])
        
        success_rate = df['success'].mean()
        average_loops = df['loops'].mean()
        average_loops_success = df.loc[df['success'], 'loops'].mean() if df['success'].any() else 0
        average_acc = df['acc'].mean()
        average_prec = df['prec'].mean()
        average_f1 = df['f1'].mean()
        is_min_rate = df['is_min'].mean()
        average_cost_rate_success = df.loc[df['success'], 'cost_rate'].mean() if df['success'].any() else 0
        
        return success_rate, average_loops, average_loops_success, average_acc, average_prec, average_f1, \
            is_min_rate, average_cost_rate_success

    def validate(self):
        """Validate all scenarios are solvable with optimal planner"""
        [scenario.validate(self.planner.oam_db, self.actor_skill_mapping) for scenario in self.scenarios]

    def evaluate_n(self, times):
        """
        Run evaluation multiple times for statistical significance
        :param times: number of evaluation runs
        :return: averaged metrics across all runs
        """
        evaluation_data = [self.evaluate() for _ in range(times)]
        df = pd.DataFrame.from_records(evaluation_data, columns=['success_rate', 'average_loops',
                                                                 'average_loops_success', 'average_acc', 'average_prec',
                                                                 'average_f1', 'is_min_rate',
                                                                 'average_cost_rate_success'])
        mean_values = df.mean()
        return tuple(mean_values.iloc[0])

    @classmethod
    def from_config(cls, config: LiberoPlannerEvaluationConfig):
        """
        Factory method to create LiberoPlannerEvaluation from configuration
        :param config: LIBERO evaluation configuration
        :return: configured LiberoPlannerEvaluation instance
        """
        # Select LLM model and interface based on config
        if config.model == ModelEnum.GPT_4:
            model = GPT_4
            llm = ChatGPTInterface(model)
        elif config.model == ModelEnum.GPT_5:
            model = GPT_5
            llm = ChatGPTInterface(model)
        elif config.model == ModelEnum.GPT_3:
            model = GPT_3
            llm = ChatGPTInterface(model)
        elif config.model == ModelEnum.GEMINI_1_5_FLASH:
            llm = GeminiInterface(GEMINI_1_5_FLASH)
        elif config.model == ModelEnum.GEMINI_1_5_FLASH_8B:
            llm = GeminiInterface(GEMINI_1_5_FLASH_8B)
        elif config.model == ModelEnum.GEMINI_1_5_PRO:
            llm = GeminiInterface(GEMINI_1_5_PRO)
        elif config.model == ModelEnum.GEMINI_2_0_FLASH:
            llm = GeminiInterface(GEMINI_2_0_FLASH)
        elif config.model == ModelEnum.GEMINI_2_0_FLASH_EXP:
            llm = GeminiInterface(GEMINI_2_0_FLASH_EXP)
        elif config.model == ModelEnum.GEMINI_2_5_FLASH:
            llm = GeminiInterface(GEMINI_2_5_FLASH)
        elif config.model == ModelEnum.GEMINI_2_5_PRO:
            llm = GeminiInterface(GEMINI_2_5_PRO)
        elif config.model == ModelEnum.GEMINI_EXP_1121:
            llm = GeminiInterface(GEMINI_EXP_1121)
        elif config.model == ModelEnum.GEMINI_EXP_1114:
            llm = GeminiInterface(GEMINI_EXP_1114)
        else:
            # Default to GPT-3
            model = GPT_3
            llm = ChatGPTInterface(model)

        autoregression = config.autoregressive == AutoregressionEnum.ON

        # Select object affordance mapping based on classes
        if config.classes == ClassesEnum.LIBERO:
            classes_file = "libero_classes.json"
            oam_file = "libero_oam.json"
        elif config.classes == ClassesEnum.SAYCAN:
            classes_file = "saycan_classes.json"
            oam_file = "saycan_oam.json"
        else:
            classes_file = "simulation_classes_.json"
            oam_file = "gpt-4_.json"

        # Select scenario file based on LIBERO suite
        if config.suite == LiberoSuiteEnum.LIBERO_GOAL:
            scenario = "libero_goal.csv"
        elif config.suite == LiberoSuiteEnum.LIBERO_10:
            scenario = "libero_10.csv"
        elif config.suite == LiberoSuiteEnum.LIBERO_90:
            scenario = "libero_90.csv"
        elif config.suite == LiberoSuiteEnum.LIBERO_SPATIAL:
            scenario = "libero_spatial.csv"
        elif config.suite == LiberoSuiteEnum.LIBERO_OBJECT:
            scenario = "libero_object.csv"
        else:
            scenario = "libero_goal.csv"  # Default

        # Set base path for LIBERO scenarios
        libero_base_path = os.path.join(os.path.dirname(__file__), "data", "libero", "scenarios")

        # Setup actor capabilities - use absolute path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        capabilities_path = os.path.join(project_root, "planning_memory", "data", "capabilities", "all_capabilities.json")
        
        # Debug: print path to ensure it's correct
        print(f"Looking for capabilities at: {capabilities_path}")
        if not os.path.exists(capabilities_path):
            print(f"Capabilities file not found, trying fallback path...")
            capabilities_path = "/Users/stefantabakov/Desktop/autogpt-p/planning_memory/data/capabilities/all_capabilities.json"
        capabilities = StaticCapabilityProvider(capabilities_path)
        capabilities.process_skills()
        actor_provider = DynamicActorProvider("robot0", "robot_profile", "robot", capabilities)
        actor_skill_mapping = ActorSkillMapping([actor_provider.get_actor()])
        
        # LLM interface already set up above based on model type
        
        # Setup object affordance mapping database
        oamdb = ObjectAffordanceMappingDatabase.load_from_data(classes_file,
                                                               "proposed_affordances_alternative.json",
                                                               oam_file)
        
        # Configure autoregression
        if autoregression:
            max_loops = 10
        else:
            max_loops = 1

        # Create planner
        planner = AutoGPTPlanner(llm, [], [], [], oamdb, actor_skill_mapping=actor_skill_mapping, 
                               max_loops=max_loops, save_pddl_files=False)

        return cls(planner, actor_skill_mapping, os.path.join(libero_base_path, scenario), 
                  config.suite.name.lower())


def write_libero_results(filename, results):
    """Write LIBERO evaluation results to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['success_rate', 'average_loops', 'average_loops_success', 'average_acc', 'average_prec',
                         'average_f1', 'is_min_rate', 'average_cost_rate_success'])
        # Write results
        writer.writerow(results)


if __name__ == "__main__":
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run LIBERO evaluation with specified model')
    parser.add_argument('model', type=str, 
                       choices=['GPT_3', 'GPT_4', 'GPT_5', 
                               'GEMINI_1_5_FLASH', 'GEMINI_1_5_FLASH_8B', 
                               'GEMINI_1_5_PRO', 'GEMINI_2_0_FLASH', 'GEMINI_2_0_FLASH_EXP',
                               'GEMINI_2_5_FLASH', 'GEMINI_2_5_PRO',
                               'GEMINI_EXP_1121', 'GEMINI_EXP_1114'],
                       help='Model to use for evaluation')
    parser.add_argument('--suite', type=str, default='LIBERO_90',
                       choices=['LIBERO_GOAL', 'LIBERO_10', 'LIBERO_90', 
                               'LIBERO_SPATIAL', 'LIBERO_OBJECT'],
                       help='LIBERO suite to evaluate (default: LIBERO_90)')
    parser.add_argument('--autoregression', type=str, default='ON',
                       choices=['ON', 'OFF'],
                       help='Use autoregression (default: ON)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Google API key for Gemini models')
    
    args = parser.parse_args()
    
    # Set Python path
    os.environ['PYTHON_3_8_16'] = sys.executable
    
    # Set Google API key for Gemini models if provided
    if args.api_key:
        os.environ['GOOGLE_API_KEY'] = args.api_key
    elif args.model.startswith('GEMINI') and not os.environ.get('GOOGLE_API_KEY'):
        print("Error: Google API key required for Gemini models")
        print("Please provide it via --api-key argument or set GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    # Configuration
    BASE_PATH = "/Users/stefantabakov/Desktop/autogpt-p"
    LIBERO_LOGS_DIR = os.path.join(BASE_PATH, "libero_logs")
    LIBERO_RESULTS_DIR = os.path.join(BASE_PATH, "libero_results")
    
    # Ensure directories exist
    os.makedirs(LIBERO_LOGS_DIR, exist_ok=True)
    os.makedirs(LIBERO_RESULTS_DIR, exist_ok=True)

    # Create configuration based on arguments
    model_enum = ModelEnum[args.model]
    suite_enum = LiberoSuiteEnum[args.suite]
    autoregression_enum = AutoregressionEnum[args.autoregression]
    
    configs = [LiberoPlannerEvaluationConfig(model_enum, autoregression_enum,
                                           ClassesEnum.LIBERO, suite_enum)]
    
    print(f"Running LIBERO evaluation with:")
    print(f"  Model: {args.model}")
    print(f"  Suite: {args.suite}")
    print(f"  Autoregression: {args.autoregression}")
    print(f"  Python: {os.environ['PYTHON_3_8_16']}")
    
    for config in configs:
        file_name = LIBERO_FILE_FORMAT.format(config.model, config.autoregressive, config.classes, config.suite)
        
        # Setup logging
        logging.basicConfig(filename=os.path.join(LIBERO_LOGS_DIR, file_name + ".log"),
                            level=logging.INFO,
                            format='%(message)s')
        
        results_file_path = os.path.join(LIBERO_RESULTS_DIR, file_name + ".csv")

        try:
            evaluation = LiberoPlannerEvaluation.from_config(config)
            result = evaluation.evaluate()
            write_libero_results(results_file_path, result)
            print(f"Completed evaluation: {config}")
            print(f"Results saved to: {results_file_path}")
        except Exception as e:
            print(f"Error in evaluation {config}: {e}")
            import traceback
            traceback.print_exc()