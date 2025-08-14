import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pddl.domain import Domain
from pddl.plan import Plan
from pddl.problem import Problem
import subprocess

FAST_DOWNWARD_VAR = "FAST_DOWNWARD_ROOT"
FAST_DOWNWARD_DIR = os.environ[FAST_DOWNWARD_VAR]
FAST_DOWNWARD_PY = "fast-downward.py"
FAST_DOWNWARD_PATH = os.path.join(FAST_DOWNWARD_DIR, FAST_DOWNWARD_PY)

PYTHON_VAR = "PYTHON_3_8_16"
PYTHON_PATH = os.environ[PYTHON_VAR]


class PDDLPlanner(ABC):
    def __init__(self, domain: Domain):
        self.domain = domain

    @abstractmethod
    def solve(self, problem: Problem):
        pass


class FastDownwardPlanner(PDDLPlanner):

    def __init__(self, domain: Domain, planner_path: str = FAST_DOWNWARD_PATH):
        super().__init__(domain)
        self.planner_path = planner_path

    def solve(self, problem: Problem) -> Plan:
        # this may be a very ugly way to do it but performance-wise it makes little difference
        # as the files are very small
        temp_folder = tempfile.mkdtemp()

        # Create file path inside the temporary folder
        domain_file_path = os.path.join(temp_folder, "domain.pddl")
        problem_file_path = os.path.join(temp_folder, "problem.pddl")
        plan_file_path = "sas_plan"

        # print(str(self.domain))
        # print(str(problem))

        try:
            # Write content to the file
            with open(domain_file_path, 'w') as file:
                file.write(self.domain.__str__())
            with open(problem_file_path, 'w') as file:
                file.write(problem.__str__())

            merge_and_shrink = "merge_and_shrink(shrink_strategy=shrink_bisimulation(greedy=false)," \
                               "merge_strategy=merge_sccs(order_of_sccs=topological," \
                               "merge_selector=score_based_filtering" \
                               "(scoring_functions=[goal_relevance,dfp,total_order]))," \
                               "label_reduction=exact(before_shrinking=true,before_merging=false)," \
                               "max_states=50k,threshold_before_merge=1)"

            # execute planner
            command = [PYTHON_PATH, self.planner_path, domain_file_path,
                       problem_file_path, '--search', "astar({})".format("ff()")]
            # subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result.check_returncode()

            # Read content of plan
            with open(plan_file_path, 'r') as file:
                content = file.read()
                plan = Plan.from_string(content, self.domain.actions, problem.objects)
                # print("Resulting plan:")
                # [print(str(a)) for a in plan.actions]
            os.remove(plan_file_path)
        except FileNotFoundError:
            plan = Plan([])
            plan.costs = -1
        except subprocess.CalledProcessError:
            plan = Plan([])
            plan.costs = -1

        finally:
            # Remove the temporary folder and all its contents
            shutil.rmtree(temp_folder)

        return plan


class FastForward(PDDLPlanner):

    def __init__(self, domain: Domain, planner_path: str):
        super().__init__(domain)
        self.planner_path = planner_path

    def solve(self, problem: Problem) -> Plan:
        # this may be a very ugly way to do it but performance-wise it makes little difference
        # as the files are very small
        temp_folder = tempfile.mkdtemp()
        # print(temp_folder)

        # Create file path inside the temporary folder
        domain_file_path = os.path.join(temp_folder, "domain.pddl")
        problem_file_path = os.path.join(temp_folder, "problem.pddl")
        plan_file_path = "sas_plan"

        # print(str(self.domain))
        # print(str(problem))

        try:
            # Write content to the file
            with open(domain_file_path, 'w') as file:
                file.write(self.domain.__str__())
            with open(problem_file_path, 'w') as file:
                file.write(problem.__str__())

            # execute planner
            command = [self.planner_path, '-p', str(temp_folder) + "/", '-o', "domain.pddl", '-f', "problem.pddl", '-s', "4"]
            # subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(command)

            # Read content of plan
            with open(plan_file_path, 'r') as file:
                content = file.read()
                plan = Plan.from_string(content, self.domain.actions, problem.objects)
                # print("Resulting plan:")
                # [print(str(a)) for a in plan.actions]
            os.remove(plan_file_path)
        except FileNotFoundError:
            plan = Plan([])

        finally:
            # Remove the temporary folder and all its contents
            shutil.rmtree(temp_folder)

        return plan

