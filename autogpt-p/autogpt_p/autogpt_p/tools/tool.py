from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from autogpt_p.state_machine.auto_gpt_p_memory import Memory


class Tool(ABC):
    """
    The abstract base class of a tool that can be used by a LLM. This class is designed to be only used by an LLM as it
    is called    with string parameters and no raw data. It is really important to make a good description of the tool
    and provide examples of the usage for the LLM to properly understand the tool.
    """

    def __init__(self, keyword: str, parameters: List[str], description: str, memory: Memory):
        self.keyword = keyword
        self.parameters = parameters
        self.description = description.format(*parameters)
        self.memory = memory
        self.executable = False

    @abstractmethod
    def get_executable(self, parameters: List[str]) -> Tool:
        """
        Returns an executable version of a generic tool. This is done to have the generic tool as a blueprint
        to instantiate multiple tools of the same type with different parameters
        :param parameters:
        :return:
        """
        pass

    def execute(self) -> str:
        """
        Executes the tool with the parameters if the tool is in executable state i.e. it was created with the
        get_executable method
        :return: a status message that indicates the result of the tool to the LLM
        This uses strings instead of raw data as the tool class is explicitly designed to be used with an LLM as
        tool user
        """
        if not self.executable:
            print("NOT EXECUTING TOOL")
            return "The tool was not in executable state"
        else:
            print("EXECUTING TOOL")
            return self._execute()

    @abstractmethod
    def abort(self):
        pass

    @abstractmethod
    def _execute(self) -> str:
        pass

    def _format_parameters(self) -> List[str]:
        return [f"{s}" for s in self.parameters]

    def __str__(self):
        return "{} {} - {}".format(self.keyword, " ".join(self._format_parameters()), self.description)


