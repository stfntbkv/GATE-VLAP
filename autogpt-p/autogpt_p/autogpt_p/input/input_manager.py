from abc import ABC, abstractmethod


class InputManager(ABC):

    def __init__(self):
        self.input = ""
        self.waiting = False

    def get_input(self) -> str:
        return self.input

    def wait_for_input(self):
        self.waiting = True
        self.fetch_input()
        self.waiting = False

    @abstractmethod
    def fetch_input(self):
        pass






