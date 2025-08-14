import logging
from typing import List

from autogpt_p.evaluation.simulated_scene import SimulatedScene
from autogpt_p.substitution.substitution import Substitution


class AlternativeSuggestionScenario:
    """

    """

    def __init__(self, scene: SimulatedScene, user_task: str, missing_object: str, alternatives: List[str]):
        """
        Creates a new substition scenario
        :param scene:
        :param user_task:
        :param missing_object:
        :param alternatives:
        """
        self.scene = scene
        self.user_task = user_task
        self.missing_object = missing_object
        self.alternatives = alternatives

    def evaluate_substitution(self, substitution: Substitution) -> bool:
        """
        Uses AutoGPTPlanner to generate a plan for the given user_task of this scenario and calculates different metrics
        :param substitution:
        :return: a tuple of whether the suggested alternative_suggestion is in the ground truth and if yes its rank, otherwise -1
        """
        print(self.user_task)
        # make sure the substitution memory is clean here
        substitution.substitution_memory.reset()
        substitution.ask_for_substitution(self.missing_object, self.user_task, self.scene.objects)
        alternative = substitution.substitution_memory.get_substitution(self.missing_object)
        print(alternative)
        print(self.alternatives)
        correct = (alternative in self.alternatives)
        print(correct)
        logging.info("Missing: {} - Substitution: {} - Desired: {}".format(self.missing_object, alternative,
                                                                           self.alternatives))
        return correct
