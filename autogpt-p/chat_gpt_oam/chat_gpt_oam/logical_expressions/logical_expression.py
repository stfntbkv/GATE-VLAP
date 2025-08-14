import os
import threading
from typing import List

import openai
from openai import OpenAI

from chat_gpt_oam.tools.rate_manager import RateManager

openai.api_key = os.environ["OPENAI_API_KEY"]
# preparation_specific = {"role": "system", "content": "You answer the questions with a step by step explanation how you came to your conclusion. End your answer with \"ANSWER:yes/no\""}
preparation_specific = {"role": "system", "content": "You answer the questions with a either yes or no without further explanation."}
preparation_summarize = {"role": "system", "content": "You are an assistant that should provide useful "
                                                      "information to the user."}
summarize_prep = "Summarize your previous answer with either yes or no"

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class LogicExpression:
    def __init__(self):
        return

    def evaluate(self):
        pass

class TimeoutException(Exception):
    pass


class AtomicExpression(LogicExpression):

    def __init__(self, phrase: str, two_step_mode=False, model='gpt-4', history=None):
        super().__init__()
        self.phrase = phrase
        self.model = model
        self.two_step_mode = two_step_mode
        self.history = history

    def evaluate(self) -> bool:
        error = True
        r = ""
        count = 0
        while len(r) < 1:
            RateManager().wait_until_rate()
            print(count)
            count += 1
            r = self.await_response()
        return "yes" in r.lower()

    def await_response(self):
        r = ""
        try:
            aff_message = {"role": "user", "content": self.phrase}
            # print(self.phrase)
            if not self.two_step_mode:
                response = client.chat.completions.create(model=self.model,
                                                          messages=[preparation_specific, aff_message],
                                                          temperature=0,
                                                          timeout=50)
            else:
                response = client.chat.completions.create(model=self.model,
                                                          messages=[preparation_summarize, aff_message],
                                                          temperature=0,
                                                          timeout=50)
                answer_to_summarize = r = response.choices[0].message.content
                print(answer_to_summarize)
                summarize_message = {"role": "user", "content": summarize_prep}
                assistant_message = {"role": "assistant", "content": answer_to_summarize}
                response = client.chat.completions.create(model=self.model,
                                                          messages=[preparation_summarize, aff_message],
                                                          temperature=0,
                                                          timeout=50)
            r = response.choices[0].message.content
            self.history.append((self.phrase, r))
        except Exception as e:
            print(str(e))
        return r


class AndExpression(LogicExpression):

    def __init__(self, expressions: List[LogicExpression]):
        super().__init__()
        self.expressions = expressions

    def evaluate(self) -> bool:
        for e in self.expressions:
            if not e.evaluate():
                return False
        return True


class OrExpression(LogicExpression):

    def __init__(self, expressions: List[LogicExpression]):
        super().__init__()
        self.expressions = expressions

    def evaluate(self) -> bool:
        for e in self.expressions:
            if e.evaluate():
                return True
        return False


class NotExpression(LogicExpression):
    def __init__(self, expression: LogicExpression):
        super().__init__()
        self.expression = expression

    def evaluate(self) -> bool:
        r = self.expression.evaluate()
        return not r


class TrueExpression(LogicExpression):

    def __init__(self):
        super().__init__()
        return

    def evaluate(self):
        return True


class FalseExpression(LogicExpression):

    def __init__(self):
        super().__init__()
        return

    def evaluate(self):
        return False
