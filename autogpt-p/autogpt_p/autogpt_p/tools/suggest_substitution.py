from typing import List

from autogpt_p.llm.llm_factory import LLMFactory
from autogpt_p.substitution.affordance_based_substituiton import AffordanceBasedSubstitution
from autogpt_p.tools.tool import Tool

KEYWORD = "SUGGEST_ALTERNATIVE"
PARAMETERS = ["missing_object"]
DESCRIPTION = "Ask the user to replace the object {} that they requested with another object with the " \
              "same function. You may only ask for substitution of one object at a time. " \
              "Example usage: SUGGEST_ALTERNATIVE spoon"


class SuggestSubstitution(Tool):
    """
    Ask the user to replace the object {} that they requested with another object with the same function.

    Memory-Reads: Object-Memory, Substitution-Memory
    Memory-Writes: Substitution-Memory

    Failure-Cases:
        - Object is in Scene
        - Object has already been substituted
        - User-Refuses substitution
    """

    def __init__(self, memory):
        super().__init__(KEYWORD, PARAMETERS, DESCRIPTION, memory)
        self.llm = LLMFactory.get_instance().produce_llm()

    def get_executable(self, parameters: List[str]) -> Tool:
        tool = SuggestSubstitution(self.memory)
        tool.executable = True
        # in case the tool is called with a list of objects
        if "," in parameters[0]:
            tool.parameters = parameters[0].split(",")[0]
        tool.parameters = parameters
        return tool

    def _execute(self):
        substitution = AffordanceBasedSubstitution(self.llm, self.memory.substitution_memory, self.memory.oam_db)
        # this method updates the substitution memory
        print("Asking for alternative_suggestion for " + str(self.parameters[0]))
        return substitution.ask_for_substitution(self.parameters[0],
                                                 self.memory.command_memory[len(self.memory.command_memory) - 1],
                                                 self.memory.objects)

    def abort(self):
        return

    def __eq__(self, other):
        return isinstance(other, SuggestSubstitution)
