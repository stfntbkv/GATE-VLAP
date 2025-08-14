from enum import Enum, auto


class ModelEnum(Enum):
    GPT_3 = auto()
    GPT_4 = auto()

    def __str__(self):
        return self.name.lower()


class MethodEnum(Enum):
    LIST_OBJECTS = auto()
    LIST_AFFORDANCES = auto()
    BINARY = auto()
    BINARY_LOGIC = auto()

    def __str__(self):
        return self.name.lower()


class AffordancesEnum(Enum):
    BASIC = auto()
    PROPOSED = auto()

    def __str__(self):
        return self.name.lower()


class EvaluationConfig:
    def __init__(self, model: ModelEnum, method: MethodEnum, affordances: AffordancesEnum):
        self.model = model
        self.method = method
        self.affordances = affordances

    def __str__(self):
        return f"Model: {self.model.name}, Method: {self.method.name}, Affordances: {self.affordances.name}"




def generate_all_combinations():
    combinations = []

    for model in ModelEnum:
        for method in MethodEnum:
            for affordance in AffordancesEnum:
                combinations.append(EvaluationConfig(model, method, affordance))

    return combinations
