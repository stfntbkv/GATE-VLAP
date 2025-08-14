from enum import Enum, auto


class ModelEnum(Enum):
    GPT_3 = auto()
    GPT_4 = auto()
    GPT_5 = auto()

    def __str__(self):
        return self.name.lower()


class ScenarioEnum(Enum):
    ALTERNATIVE = auto()
    EXPLORATION = auto()
    PARTIAL = auto()
    PLAN = auto()
    COMPLEX = auto()

    def __str__(self):
        return self.name.lower()


class AutoGPTPEvaluationConfig:
    def __init__(self, model: ModelEnum, scenario: ScenarioEnum):
        self.model = model
        self.scenario = scenario

    def __str__(self):
        return f"Model: {self.model.name}, Affordances: {self.scenario.name}"


def generate_all_combinations():
    combinations = []

    for model in ModelEnum:
        for scenario in ScenarioEnum:
            combinations.append(AutoGPTPEvaluationConfig(model, scenario))

    return combinations
