from abc import ABC

from autogpt_p.input.input_manager import InputManager


class ConsoleInputManager(InputManager, ABC):

    def fetch_input(self):
        self.input = input("Please enter a command: ")
        print(self)