import itertools
import logging
from typing import List

from object_affordance_mapping.object_affordance_mapping import ObjectAffordanceMappingDatabase, AffordanceType
from object_detection.detection_memory_segment import DetectedObject

from autogpt_p.llm.llm_interface import LLMInterface
from autogpt_p.substitution.naive_substitution import NaiveSubstitution
from autogpt_p.substitution.substitution import Substitution, extract_list, extract_object
from autogpt_p.substitution.substitution_memory import SubstitutionMemory

RELEVANT_AFFORDANCES_PROMPT = "Out of the following affordances for a {}, which ones does the user need for the " \
                              "request \"{}\"? The affordances are:\n{}\n " \
                              "End your answer with listing the affordances as strings in a python list format"

REPLACEMENT_PROMPT = "Which of these objects can replace a {} for the affordance {} the best. The object(s) are: {} \n " \
                     "Finish your answer by naming exactly one object with OBJECT:<object>"

CORRECTION_PROMPT = "The object {} is not within the given list. Please chose an object from the given list:\n{}"


class AffordanceBasedSubstitution(Substitution):

    def __init__(self, llm: LLMInterface, substitution_memory: SubstitutionMemory,
                 oam_db: ObjectAffordanceMappingDatabase, ask_for_confirmation=True):
        """
        Initializes a new Substitution task.
        :param llm: the interface to a large language model. the llm should be fresh and not reused after this task
        because the history of the conversation is saved by the llm
        :param substitution_memory: the memory where the collected allowed_substitutions are written to
        :param oam_db: the object affordance mapping that is used to filter the relevant objects
        """
        super().__init__(llm, substitution_memory, ask_for_confirmation)
        self.oam_db = oam_db

    def get_substitution(self, object_class: str, prompt: str, objects: List[DetectedObject]):
        task = prompt
        object_affordances = self.oam_db.get_affordance_by_object_name(object_class)
        prompt = RELEVANT_AFFORDANCES_PROMPT.format(object_class, prompt,
                                                    "\n".join(
                                                        [a.name + ": " + a.description for a in object_affordances]))
        response = self.llm.prompt(prompt, True)
        logging.info("Q:{}\nA: {}\n-----------------------------------------\n".format(prompt, response))
        print("Q:{}\nA: {}\n-----------------------------------------\n".format(prompt, response))
        relevant_affordances_names = extract_list(response)
        relevant_affordances = [self.oam_db.oa_database.get_affordance_by_name(name) for name in
                                relevant_affordances_names]
        # filter out all affordances that have not matching object
        relevant_affordances = [a for a in relevant_affordances if
                                len(self.oam_db.get_objects_with_affordance(a)) > 0]
        candidate_objects = self._filter_objects(objects, relevant_affordances)

        if len(candidate_objects) == 1:
            return candidate_objects[0].class_name

        # fallback strategy if no candidate object exists
        if len(candidate_objects) == 0:
            self.llm.reset_history()
            return NaiveSubstitution(self.llm, self.substitution_memory, self.ask_for_confirmation) \
                .get_substitution(object_class, task, objects)

        rarest_affordance = self._get_rarest_affordance(objects, relevant_affordances)

        # fallback strategy if no affordance matches
        if rarest_affordance is None:
            self.llm.reset_history()
            return NaiveSubstitution(self.llm, self.substitution_memory, self.ask_for_confirmation) \
                .get_substitution(object_class, task, objects)

        candidate_object_names = list(set([o.class_name for o in candidate_objects]))

        prompt = REPLACEMENT_PROMPT.format(object_class, rarest_affordance.name,
                                           ",".join(candidate_object_names))

        response = self.llm.prompt(prompt, True)
        logging.info("Q:{}\nA: {}\n-----------------------------------------\n".format(prompt, response))
        print("Q:{}\nA: {}\n-----------------------------------------\n".format(prompt, response))
        alternative = extract_object(response)
        counter = 0

        while alternative not in candidate_object_names and counter < 3:
            counter += 1
            prompt = CORRECTION_PROMPT.format(alternative, ", ".join(candidate_object_names))
            response = self.llm.prompt(prompt, True)
            logging.info("Q:{}\nA: {}\n-----------------------------------------\n".format(prompt, response))
            print("Q:{}\nA: {}\n-----------------------------------------\n".format(prompt, response))
            alternative = extract_object(response)

        self.llm.reset_history()
        return alternative

    def _filter_objects(self, objects: List[DetectedObject],
                        relevant_affordances: List[AffordanceType]) -> List[DetectedObject]:
        return list(filter(lambda o: all(item in self.oam_db.get_affordance_by_object_name(o.class_name)
                                         for item in relevant_affordances), objects))

    def _get_rarest_affordance(self, objects: List[DetectedObject],
                               relevant_affordances: List[AffordanceType]) -> AffordanceType:
        affordance_list = list(itertools.chain(
            *[self.oam_db.get_affordance_by_object_name(o.class_name) for o in objects]))
        affordance_counter = {a: 0 for a in relevant_affordances}
        for a in affordance_list:
            if a in relevant_affordances:
                affordance_counter[a] = affordance_counter[a] + 1
        affordance_counter = {k: v for k, v in affordance_counter.items() if v > 0}
        return min(affordance_counter, key=affordance_counter.get) if len(affordance_counter) > 0 else None
