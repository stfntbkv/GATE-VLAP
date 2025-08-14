from __future__ import annotations

from autogpt_p.helpers.singleton import Singleton
from autogpt_p.input.console_input_manager import ConsoleInputManager
from autogpt_p.input.input_manager import InputManager
from autogpt_p.input.simulated_user import SimulatedUser

COMMAND_LINE = "CL"
SPEECH = "SPEECH"


class InputManagerFactory(Singleton):
    _instance = None

    @classmethod
    def get_instance(cls) -> InputManagerFactory:
        return cls._instance

    def __init__(self, type: str):
        self.type = type
        self.dummy = None

    def set_dummy(self, dummy: SimulatedUser):
        self.dummy = dummy

    def produce_input_manager(self) -> InputManager:
        if self.dummy:
            return self.dummy
        if self.type == COMMAND_LINE:
            return ConsoleInputManager()
        else:
            return ConsoleInputManager()
