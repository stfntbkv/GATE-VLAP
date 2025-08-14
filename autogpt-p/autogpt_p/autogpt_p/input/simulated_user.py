import csv
from abc import ABC
from typing import Tuple, List

from autogpt_p.input.input_manager import InputManager


class PrinterMemory:

    def __init__(self):
        self.last_line = ""

    def print(self, string: str):
        print(str)
        self.last_line = string


printer = PrinterMemory()


def extract_substitution(string: str) -> Tuple[str, str]:
    begin = len("Is the substitution ")
    end = string.find(" sufficient")
    args = string[begin:end].split("->")
    return args[0].strip(), args[1].strip()


class SimulatedUser(InputManager, ABC):

    def __init__(self, allowed_substitutions: List[Tuple[str, str]]):
        super().__init__()
        self.allowed_substitutions = allowed_substitutions

    def fetch_input(self):
        missing, substitution = extract_substitution(printer.last_line)
        self.input = "yes" if (missing, substitution) in self.allowed_substitutions else "no"

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            allowed_substitutions = [(row["missing"], row["substitution"]) for row in reader]
            return SimulatedUser(allowed_substitutions)
