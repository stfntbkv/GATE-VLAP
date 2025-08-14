import logging
from abc import ABC

from autogpt_p.llm.llm_interface import LLMInterface
from openai import OpenAI

client = OpenAI()


USER_ROLE = "user"
CONTENT = "content"
ASSISTANT_ROLE = "assistant"
MESSAGE = "message"
CHOICES = "choices"
ROLE = "role"

GPT_3 = "gpt-3.5-turbo-16k"
GPT_4 = "gpt-4-0613"
# GPT_4 = "gpt-4-turbo"
GPT_5 = "gpt-5"


class ChatGPTInterface(LLMInterface, ABC):

    def __init__(self, model=GPT_4, history=None):
        super().__init__(history)
        self.model = model

    def branch(self):
        """
        Branches the history of the chat out by returning a clone of the history
        :return:
        """
        return ChatGPTInterface(self.model, self.history)

    def prompt(self, message: str, add_to_history=True) -> str:
        """
        Convinience method to access chatgpt api and optionally add messages to history
        :param message:
        :param add_to_history: if true the message and the answer will be added to the history and subsequently
        added to all api calls (which raises the token so beware the length of the history
        :return:
        """
        prompt = {ROLE: USER_ROLE, CONTENT: message}
        dialogue = self.history + [prompt]
        error = True
        r = ""
        logging.info(message)
        # ChatGPT Interaction
        while error:
            try:
                response = client.chat.completions.create(model=self.model,
                messages=dialogue,
                temperature=1,
                #reasoning_effort="high", 
                timeout=45)
                r = response.choices[0].message.content
                logging.debug(r)
                error = False
            except Exception as e:
                logging.info(str(e))
        logging.info(r)
        if add_to_history:
            response = {ROLE: ASSISTANT_ROLE, CONTENT: r}
            self.history = dialogue + [response]
        return r
