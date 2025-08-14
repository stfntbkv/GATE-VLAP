from typing import Dict, List

import numpy as np


class ExplorationMemory:

    def __init__(self, explorable_locations: Dict[str, np.ndarray], current_location: str):
        self.explorable_locations = explorable_locations
        self.current_location = current_location
        self.explored = None
        if len(explorable_locations) > 0:
            self.reset()

    def get_explored(self) -> List[str]:
        return [loc for loc, explored in zip(list(self.explorable_locations.keys()), self.explored) if explored]

    def get_unexplored(self) -> List[str]:
        return [loc for loc, explored in zip(list(self.explorable_locations.keys()), self.explored) if not explored]

    def update_explored(self, location: str):
        self.current_location = location
        idx = list(self.explorable_locations.keys()).index(location)
        self.explored[idx] = True

    def is_explored(self, location: str) -> bool:
        idx = list(self.explorable_locations.keys()).index(location)
        return self.explored[idx]

    def get_coordinates(self, location: str) -> np.ndarray:
        return self.explorable_locations[location]

    def reset(self):
        self.explored = [False for i in range(len(self.explorable_locations.keys()))]
        self.update_explored(self.current_location)

    def __str__(self) -> str:
        return "Unexplored Locations: " + ",".join(self.get_unexplored())

