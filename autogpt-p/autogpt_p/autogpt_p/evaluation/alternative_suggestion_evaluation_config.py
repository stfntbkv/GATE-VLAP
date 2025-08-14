from enum import Enum, auto


class ModelEnum(Enum):
    GPT_3 = auto()
    GPT_4 = auto()

    def __str__(self):
        return self.name.lower()


class MethodEnum(Enum):
    NAIVE = auto()
    AFFORDANCE = auto()

    def __str__(self):
        return self.name.lower()


class ScenarioEnum(Enum):
    SIMPLE = auto()
    MEDIUM = auto()
    COMPLEX = auto()

    def __str__(self):
        return self.name.lower()


class AlternativeSuggestionEvaluationConfig:
    def __init__(self, model: ModelEnum, method: MethodEnum, scenario: ScenarioEnum):
        self.model = model
        self.method = method
        self.scenario = scenario

    def __str__(self):
        return f"Model: {self.model.name}, Method: {self.method.name}, Affordances: {self.scenario.name}"


def generate_all_combinations():
    combinations = []

    for model in ModelEnum:
        for method in MethodEnum:
            for affordance in ScenarioEnum:
                combinations.append(AlternativeSuggestionEvaluationConfig(model, method, affordance))

    return combinations
