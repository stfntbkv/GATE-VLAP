from enum import Enum, auto


class ModelEnum(Enum):
    GPT_3 = auto()
    GPT_4 = auto()
    GPT_5 = auto()
    GEMINI_1_5_FLASH = auto()
    GEMINI_1_5_FLASH_8B = auto()
    GEMINI_1_5_PRO = auto()
    GEMINI_2_0_FLASH_EXP = auto()

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
    LIBERO = auto()  # LIBERO-specific object types and affordances

    def __str__(self):
        return self.name.lower()


class LiberoSuiteEnum(Enum):
    LIBERO_GOAL = auto()    # 10 basic goal-based tasks
    LIBERO_10 = auto()      # 10 complex multi-step tasks
    LIBERO_90 = auto()      # 90 varied complexity tasks
    LIBERO_SPATIAL = auto() # 10 spatial reasoning tasks
    LIBERO_OBJECT = auto()  # 10 object manipulation tasks

    def __str__(self):
        return self.name.lower()


class LiberoPlannerEvaluationConfig:
    def __init__(self, model: ModelEnum, autoregressive: AutoregressionEnum,
                 classes: ClassesEnum, suite: LiberoSuiteEnum):
        self.model = model
        self.autoregressive = autoregressive
        self.classes = classes
        self.suite = suite

    def __str__(self):
        return f"Model: {self.model.name}, Method: {self.autoregressive.name}, Classes: {self.classes.name}, Suite: {self.suite.name}"


def generate_all_libero_combinations():
    """Generate all possible combinations of LIBERO evaluation configurations"""
    combinations = []

    for model in ModelEnum:
        for method in AutoregressionEnum:
            for classes in ClassesEnum:
                for suite in LiberoSuiteEnum:
                    combinations.append(LiberoPlannerEvaluationConfig(model, method, classes, suite))

    return combinations


def generate_libero_only_combinations():
    """Generate combinations specifically for LIBERO evaluation"""
    combinations = []

    for model in ModelEnum:
        for method in AutoregressionEnum:
            for suite in LiberoSuiteEnum:
                # Only use LIBERO classes for LIBERO evaluation
                combinations.append(LiberoPlannerEvaluationConfig(model, method, ClassesEnum.LIBERO, suite))

    return combinations