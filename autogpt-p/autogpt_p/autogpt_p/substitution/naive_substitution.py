import logging
from typing import List

from object_detection.detection_memory_segment import DetectedObject

from autogpt_p.llm.llm_interface import LLMInterface
from autogpt_p.substitution.substitution import Substitution, extract_object
from autogpt_p.substitution.substitution_memory import SubstitutionMemory

REPLACEMENT_PROMPT = "Which of these objects can replace a {} the best for the following task stated by the user.\n{" \
                     "}\nThe object(s) are: {} \n " \
                     "Finish your answer by naming exactly one object with OBJECT:<object>"


class NaiveSubstitution(Substitution):
    """
    Naive substitution tool that just asks the large language model which object replaces the given object the best.
    It does not respect the affordances.
    """

    def __init__(self, llm: LLMInterface, substitution_memory: SubstitutionMemory, ask_for_confirmation=True):
        """
        Initializes a new Substitution task.
        :param llm: the interface to a large language model. the llm should be fresh and not reused after this task
        because the history of the conversation is saved by the llm
        :param substitution_memory: the memory where the collected allowed_substitutions are written to
        """
        super().__init__(llm, substitution_memory, ask_for_confirmation)

    def get_substitution(self, object_class: str, prompt: str, objects: List[DetectedObject]) -> str:

        prompt = REPLACEMENT_PROMPT.format(object_class, prompt,
                                  ",".join(list(set([o.class_name for o in objects]))))
        response = self.llm.prompt(prompt, False)
        logging.info("Q:{}\nA: {}".format(prompt, response))
        print("Q:{}\nA: {}\n-----------------------------------------\n".format(prompt, response))
        return extract_object(response)
