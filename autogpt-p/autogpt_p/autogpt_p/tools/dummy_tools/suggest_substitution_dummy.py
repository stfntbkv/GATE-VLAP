from autogpt_p.tools.suggest_substitution import SuggestSubstitution


class SuggestSubstitutionDummy(SuggestSubstitution):

    def __init__(self, memory, substitution):
        super().__init__(memory)
        self.substitutions = substitution

    def execute(self):
        self.memory.substitution_memory.add_substitution(self.parameters[0], self.substitutions[self.parameters[0]])
