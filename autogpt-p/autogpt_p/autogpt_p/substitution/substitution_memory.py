class SubstitutionMemory:
    """
    Memory containing possible allowed_substitutions for object classes where the object classes are referenced by a
    string representation
    """

    def __init__(self):
        """
        Creates a new empty substitution memory.
        """
        self.substitutions = {}

    def add_substitution(self, object_name: str, alternative_name: str):
        """
        This method allows you to map an alternative_suggestion name to an existing object name within the instance's
        allowed_substitutions dictionary.

        :param object_name: The original name of the object for which we are providing an alternative_suggestion. This must be
        of type str.
        :param alternative_name: The alternative_suggestion name that will be mapped to the object_name in the
        allowed_substitutions dictionary. This must also be of type str.

        :return: None. The method updates the instance's allowed_substitutions dictionary in-place
        , but does not return anything.
    """
        self.substitutions[object_name] = alternative_name

    def get_substitution(self, object_name: str) -> str:
        """
        This method retrieves the alternative_suggestion name for a given object name from the instance's allowed_substitutions
        dictionary.

        :param object_name: The original name of the object for which we are retrieving an alternative_suggestion. This must be
        of type str.

        :return: The alternative_suggestion name associated with the provided object_name in the allowed_substitutions dictionary.
        If no alternative_suggestion name exists, it will raise a KeyError.
        """
        return self.substitutions[object_name]

    def has_substitution(self, object_name: str):
        """
        This method checks if a given object name has an alternative_suggestion name in the instance's allowed_substitutions
        dictionary.

        :param object_name: The original name of the object for which we are checking for an alternative_suggestion. This must
        be of type str.

        :return: Returns True if an alternative_suggestion name exists for the provided object_name in the allowed_substitutions
        dictionary, and False otherwise.
        """
        return object_name in self.substitutions.keys()

    def get_substitution_chain_ends(self):
        """
        This method retrieves all unique alternative_suggestion names from the instance's allowed_substitutions dictionary.

        :return: A set of all unique alternative_suggestion names from the allowed_substitutions dictionary. This can be useful for
        finding the end of a substitution chain, as these are the names that do not map to any other names.
        """
        self._shorten_substitution_chain()
        return set(self.substitutions.values())

    def substitute_in_prompt(self, prompt: str, shorten=True) -> str:
        """
        This method substitutes all occurrences of the object names in the given prompt with their respective
        alternative_suggestion names from the instance's allowed_substitutions dictionary.

        :param prompt: The input string in which to replace object names with their alternative_suggestion names. This must be
        of type str.
        :param shorten: A boolean that determines whether to shorten the substitution chain before
        performing replacements. If True, the _shorten_substitution_chain() method is called before
        allowed_substitutions are made. Default is True.

        :return: The input prompt, but with all instances of each object name replaced with their corresponding
        alternative_suggestion name from the allowed_substitutions dictionary.
        """
        if shorten:
            self._shorten_substitution_chain()
        # we do this so that if an object name is a part of another objects name like cup and coffee_cup
        # if the longest one is first that cannot happen
        sorted_items = sorted(self.substitutions.items(), key=lambda item: -len(item[0]))
        for object_to_replace, alternative in sorted_items:
            prompt = prompt.replace(object_to_replace, alternative)
        return prompt

    def reset(self):
        """
        Deletes all allowed_substitutions currently in memory.
        """
        self.substitutions = {}

    def _shorten_substitution_chain(self):
        new_substitutions = {}
        for key, value in self.substitutions.items():
            new_substitutions[key] = self._follow_substitution(key, value)
        self.substitutions = new_substitutions

    def _follow_substitution(self, object_name, alternative_name):
        if alternative_name in self.substitutions.keys():
            return self._follow_substitution(object_name, self.substitutions[alternative_name])
        else:
            return alternative_name
