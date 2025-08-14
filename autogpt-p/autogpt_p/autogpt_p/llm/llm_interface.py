import logging
from abc import ABC, abstractmethod

from pddl.domain import Domain
from pddl.problem import Problem
from pddl.problem import *
from autogpt_p.planning.goal_validator import ValidationError
from autogpt_p.incremental_goal_memory.incremental_goal_memory import \
    IncrementalGoalMemory
from autogpt_p.helpers.tokens import count_tokens


PLANNING_CONTEXT = "You are the humanoid service robot robotXYZ assisting a human in a kitchen. Your task is to turn the user " \
                   "request into a goal state in PDDL. The predicates of the domain are defined as follows:\n" \
                   "{} \n and the problem without the goal is:\n {} \n The user asks \"{}\". Explain step by step " \
                   "how you concluded that this is the desired goal state. Express the goal state in correct PDDL Syntax" \
                   "without using the ? operator. You are allowed to use disjunctions\n" \
                   # "Example: Bring me a banana -> (:goal and(inhand banana human))"
# "Here are the k closest similar requests already known, with their goal states as an example." \
PARTIAL_PLANNING_CONTEXT = "You are the humanoid service robot robotXYZ assisting a human in a kitchen. Your task is to turn the user " \
                   "request into a goal state in PDDL. If an object requested by the user does not exist just ignore it." \
                   "If an utensil needed to achieve a desired object state is not in the scene do not use that state " \
                   "in the goal. For example do not use chopped if there is nothing to cut in the scene" \
                   "If the state needs more than {} predicates, formulate a partial goal using less than {} predicates." \
                   "The predicates of the domain are defined as follows:\n" \
                   "{} \n and the problem without the goal is:\n {} \n The user(human) asks: \"{}\". Explain step by step " \
                   "how you concluded that this is the desired goal state. Express the goal state in correct PDDL Syntax" \
                   "without using the ? operator. \n" \


SENTIMENT_CONTEXT = "Your goal is to classify the sentiment of the following user statement. " \
                    "If its affirmative answer with yes if not answer with no\n " \
                    "User statement: {}"
YES = "yes"
GOAL = ":goal"


class NoGoalException(Exception):
    def __init__(self):
        pass


class LLMInterface(ABC):

    def __init__(self, history=None):
        if history:
            self.history = history
        else:
            self.history = []

    def establish_context(self):
        pass

    @abstractmethod
    def prompt(self, message: str, add_to_history=True) -> str:
        pass

    @abstractmethod
    def branch(self):
        pass

    def reset_history(self):
        self.history = []

    def ask_for_goal(self, user_request: str, domain: Domain, problem: Problem) -> str:
        predicates, indent = domain.write_predicates()
        indent.format(predicates)
        message = PLANNING_CONTEXT.format(predicates, problem.write(False), user_request)
        answer = self.prompt(message)
        if not contains_goal(answer):
            raise NoGoalException()
        return extract_goal(answer)

    def ask_for_goal_in_context_learning(self, user_request: str, domain: Domain, problem: Problem, number_of_examples) -> str:
        predicates, indent = domain.write_predicates()
        indent.format(predicates)
        memory = IncrementalGoalMemory('../incremental_goal_memory/known_pairs.csv')
        examples = memory.k_closest_requests_as_str(number_of_examples, user_request)

        message = PLANNING_CONTEXT.format(predicates, problem.write(False), user_request) + examples
        print(message)
        # amend message with examples
        answer = self.prompt(message)
        total_tokens = count_tokens(message) + count_tokens(answer)
        logging.info("TOTAL TOKENS BY THIS REQUEST")
        logging.info(total_tokens)
        if not contains_goal(answer):
            raise NoGoalException()
        return extract_goal(answer)

    def ask_for_partial_goal(self, user_request: str, domain: Domain, problem: Problem, max_predicates=5):
        predicates, indent = domain.write_predicates()
        indent.format(predicates)
        message = PARTIAL_PLANNING_CONTEXT.format(max_predicates, max_predicates,
                                                  predicates, problem.write(False), user_request)
        answer = self.prompt(message)
        if not contains_goal(answer):
            raise NoGoalException()
        return extract_goal(answer)

    def ask_for_partial_goal_in_context_learning(self, user_request: str, domain: Domain, problem: Problem, max_predicates=5, number_of_examples=3):
        predicates, indent = domain.write_predicates()
        indent.format(predicates)
        memory = IncrementalGoalMemory('../incremental_goal_memory/known_pairs.csv')
        examples = memory.k_closest_requests_as_str(number_of_examples, user_request)

        message = PARTIAL_PLANNING_CONTEXT.format(max_predicates, max_predicates,
                                                  predicates, problem.write(False), user_request)  + examples
        print(message)
        answer = self.prompt(message)
        if not contains_goal(answer):
            raise NoGoalException()
        return extract_goal(answer)

    def correct_error(self, error: ValidationError):
        answer = self.prompt(error.print_error_message())
        if not contains_goal(answer):
            raise NoGoalException()
        return extract_goal(answer)

    def classify_sentiment(self, user_request) -> bool:
        response = self.prompt(SENTIMENT_CONTEXT.format(user_request))
        if YES in response.lower():
            return True
        return False


def extract_goal(string: str):
    start = string.find(GOAL)
    goal = ""
    while start >= 0:
        end = find_closing_parenthesis(string, start)
        new_goal = string[start + len(GOAL):end]
        goal = new_goal if len(new_goal) > len(goal) else goal
        start = string.find(GOAL, end)
    return goal


def contains_goal(string: str):
    return GOAL in string


def find_closing_parenthesis(string, open_index):
    count = 1
    for i in range(open_index + 1, len(string)):
        if string[i] == "(":
            count += 1
        elif string[i] == ")":
            count -= 1
            if count == 0:
                return i
    return -1

if __name__ == "__main__":
    # Beispiele printen und testen
    pass
