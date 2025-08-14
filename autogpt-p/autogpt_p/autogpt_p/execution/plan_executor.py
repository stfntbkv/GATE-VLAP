from abc import ABC, abstractmethod
from typing import List

from pddl.core import Type, Object
from pddl.plan import Plan, ExecutedAction
from autogpt_p.execution.pddl_scenario import ROBOT_TYPE

INTRO = "You need to do the following actions:"
OUTRO = "Confirm when you are done"


def is_robot(actor: Type) -> bool:
    return ROBOT_TYPE.name in [type.name for type in actor.get_all_supertypes()] or actor.name == ROBOT_TYPE.name


def extract_actors(action: ExecutedAction) -> List[Object]:
    # there is no checking here since if plan is a valid plan and contains only right actions there will be an
    # actor specified in the action
    return [a for a in action.get_actors()]


class PlanExecutorInterface(ABC):

    def __init__(self):
        self.plan = Plan([])

    def execute(self, plan: Plan):
        self.plan = plan
        self._execute(plan)

    @abstractmethod
    def _execute(self, plan: Plan):
        pass

    @abstractmethod
    def abort(self):
        pass

    def get_last_executed_plan(self) -> Plan:
        return self.plan

# NOTE to reviewers: this file was shortened compared to the original due to references to specific
# software that could impede double blindness
