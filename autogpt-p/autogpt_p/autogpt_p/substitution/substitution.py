import ast
from abc import ABC, abstractmethod
from typing import List
import re

from object_detection.detection_memory_segment import DetectedObject

from autogpt_p.input.input_manager_factory import InputManagerFactory
from autogpt_p.input.simulated_user import printer
from autogpt_p.llm.llm_interface import LLMInterface
from autogpt_p.substitution.substitution_memory import SubstitutionMemory

OBJECT_KEYWORD = "OBJECT:"


def extract_list(text):
    # use a regular expression to find the list-like substring
    list_string = re.search(r'\[.*?\]', text).group()

    # use ast.literal_eval to convert the string back to a list
    return ast.literal_eval(list_string)


def extract_object(text: str):
    return text[(text.find(OBJECT_KEYWORD) + len(OBJECT_KEYWORD)):].strip().replace(".", "")


class Substitution(ABC):
    """
    This class represents the communication with the user for substituting objects
    Algorithm:
    1) Based on the last user request the llm is asked which of the objects affordances is needed by the user
    2) Only keep objects in the scene that have the same affordances
    3) For the affordance that is the least represented in the scene. Aks which of the objects can substitute the
       given object
    """

    def __init__(self, llm: LLMInterface, substitution_memory: SubstitutionMemory, ask_for_confirmation=True):
        """
        Initializes a new Substitution task.
        :param llm: the interface to a large language model. the llm should be fresh and not reused after this task
        because the history of the conversation is saved by the llm
        :param substitution_memory: the memory where the collected allowed_substitutions are written to
        """
        self.llm = llm
        self.substitution_memory = substitution_memory
        self.ask_for_confirmation = ask_for_confirmation

    def ask_for_substitution(self, object_class: str, prompt: str, objects: List[DetectedObject]):
        """
        Ask the LLM for substitution the object class in context with the scene and the task
        :param object_class: the object_class to be replaced
        :param prompt: the context of the user tasks
        :param objects: the list of objects in the scene
        """
        substitution = self.get_substitution(object_class, prompt, objects)
        if self.ask_for_confirmation:
            printer.print("Is the substitution {}->{} sufficient".format(object_class, substitution))
            input = InputManagerFactory.get_instance().produce_input_manager()
            input.wait_for_input()
            user_statement = input.get_input()
            if user_statement.lower() == "yes" or self.llm.classify_sentiment(user_statement):
                self.substitution_memory.add_substitution(object_class, substitution)
                return "New Substitution: {}->{}".format(object_class, substitution)
            else:
                return "Substitution {}->{} rejected".format(object_class, substitution)
        else:
            self.substitution_memory.add_substitution(object_class, substitution)
            return "New Substitution: {}->{}".format(object_class, substitution)

    @abstractmethod
    def get_substitution(self, object_class: str, prompt: str, objects: List[DetectedObject]) -> str:
        pass
