from enum import Enum, auto


class ModelEnum(Enum):
    GPT_3 = auto()
    GPT_4 = auto()
    GPT_5 = auto()

    def __str__(self):
        return self.name.lower()


class AutoregressionEnum(Enum):
    ON = auto()
    OFF = auto()

    def __str__(self):
        return self.name.lower()


class ClassesEnum(Enum):
    SAYCAN = auto()
    SIMULATION = auto()

    def __str__(self):
        return self.name.lower()


class ScenarioEnum(Enum):
    SAYCAN_NL_PRIMITIVE = auto()
    SAYCAN_NL_VERB = auto()
    SAYCAN_NL_NOUN = auto()
    SAYCAN_STRUCTURED_LANGUAGE = auto()
    SAYCAN_EMBODIMENT = auto()
    SAYCAN_CROWD_SOURCED = auto()
    SAYCAN_LONG_HORIZON = auto()
    SAYCAN_DRAWER = auto()
    SIMPLE_TASK = auto()
    SIMPLE_GOAL = auto()
    COMPLEX_GOAL = auto()
    IMPLICIT = auto()
    KNOWLEDGE = auto()

    def __str__(self):
        return self.name.lower()


class PlannerEvaluationConfig:
    def __init__(self, model: ModelEnum, autoregressive: AutoregressionEnum,
                 classes: ClassesEnum, scenario: ScenarioEnum):
        self.model = model
        self.autoregressive = autoregressive
        self.classes = classes
        self.scenario = scenario

    def __str__(self):
        return f"Model: {self.model.name}, Method: {self.autoregressive.name}, Affordances: {self.scenario.name}"


def generate_all_combinations():
    combinations = []

    for model in ModelEnum:
        for method in AutoregressionEnum:
            for c in ClassesEnum:
                for scenario in ScenarioEnum:
                    combinations.append(PlannerEvaluationConfig(model, method, c, scenario))

    return combinations
