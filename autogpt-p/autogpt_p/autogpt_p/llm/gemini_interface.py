import logging
import os
from abc import ABC
import google.generativeai as genai

from autogpt_p.llm.llm_interface import LLMInterface

USER_ROLE = "user"
MODEL_ROLE = "model"

# Gemini 1.5 models (stable)
GEMINI_1_5_FLASH = "gemini-1.5-flash"
GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b"
GEMINI_1_5_PRO = "gemini-1.5-pro"

# Gemini 2.0 models
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"

# Gemini 2.5 models (latest with thinking capabilities)
GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_2_5_PRO = "gemini-2.5-pro"

# Experimental models
GEMINI_EXP_1121 = "gemini-exp-1121"  # Powerful experimental model
GEMINI_EXP_1114 = "gemini-exp-1114"  # Another experimental model


class GeminiInterface(LLMInterface, ABC):
    
    def __init__(self, model=GEMINI_1_5_FLASH, history=None, api_key=None):
        super().__init__(history)
        self.model_name = model
        
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        else:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
        )
        
        self.chat_session = None
        self._initialize_chat()
    
    def _initialize_chat(self):
        chat_history = []
        for msg in self.history:
            if "role" in msg and "content" in msg:
                if msg["role"] == "user":
                    chat_history.append({"role": USER_ROLE, "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": MODEL_ROLE, "parts": [msg["content"]]})
        
        self.chat_session = self.model.start_chat(history=chat_history)
    
    def branch(self):
        """
        Branches the history of the chat out by returning a clone of the history
        :return:
        """
        return GeminiInterface(self.model_name, self.history)
    
    def prompt(self, message: str, add_to_history=True) -> str:
        """
        Convenience method to access Gemini API and optionally add messages to history
        :param message:
        :param add_to_history: if true the message and the answer will be added to the history
        :return:
        """
        logging.info(message)
        
        error = True
        response_text = ""
        retry_count = 0
        max_retries = 5
        
        while error and retry_count < max_retries:
            try:
                response = self.chat_session.send_message(message)
                response_text = response.text
                logging.debug(response_text)
                error = False
            except Exception as e:
                retry_count += 1
                logging.info(f"Error (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    import time
                    time.sleep(5)
                else:
                    # Fall back to generate_content if chat fails
                    try:
                        response = self.model.generate_content(message)
                        response_text = response.text
                        error = False
                    except:
                        raise e
        
        logging.info(response_text)
        
        if add_to_history:
            user_msg = {"role": "user", "content": message}
            assistant_msg = {"role": "assistant", "content": response_text}
            self.history.append(user_msg)
            self.history.append(assistant_msg)
        
        return response_text