from __future__ import annotations

from autogpt_p.helpers.singleton import Singleton
from autogpt_p.llm.chat_gpt_interface import ChatGPTInterface, GPT_4, GPT_3, GPT_5
from autogpt_p.llm.gemini_interface import GeminiInterface, GEMINI_1_5_FLASH, GEMINI_1_5_PRO, GEMINI_2_0_FLASH_EXP, GEMINI_1_5_FLASH_8B
from autogpt_p.llm.llm_interface import LLMInterface

GPT = "GPT"
GEMINI = "GEMINI"


class LLMFactory(Singleton):

    _instance = None

    @classmethod
    def get_instance(cls) -> LLMFactory:
        return cls._instance

    def __init__(self, llm_type: str, version=""):
        self.llm_type = llm_type
        self.version = version

    def produce_llm(self) -> LLMInterface:
        if self.llm_type == GPT:
            if self.version == "5":
                return ChatGPTInterface(GPT_5)
            elif self.version == "4":
                return ChatGPTInterface(GPT_4)
            else:
                return ChatGPTInterface(GPT_3)
        elif self.llm_type == GEMINI:
            if self.version == "1.5-flash":
                return GeminiInterface(GEMINI_1_5_FLASH)
            elif self.version == "1.5-flash-8b":
                return GeminiInterface(GEMINI_1_5_FLASH_8B)
            elif self.version == "1.5-pro":
                return GeminiInterface(GEMINI_1_5_PRO)
            elif self.version == "2.0-flash-exp":
                return GeminiInterface(GEMINI_2_0_FLASH_EXP)
            else:
                return GeminiInterface(GEMINI_1_5_FLASH)
