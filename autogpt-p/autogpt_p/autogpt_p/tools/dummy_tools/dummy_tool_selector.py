from typing import List, Optional

from autogpt_p.tools.tool import Tool
from autogpt_p.tools.tool_selector import ToolSelector


class DummyToolSelector(ToolSelector):
    """
    Dummy tool selector can be used to let the tools return specific outputs
    For each type of tool there is a FIFO queue that repeats the last entry of that tool indefinitely
    For example for simulation purposes the DummyExplore tool can be used with
    the real alternative_suggestion and plan tool
    If only the correct calling of tools needs to be detected, everything can be made a dummy tool
    """

    def __init__(self, tools: List[Tool]):
        super().__init__(tools)

    def get_tool(self, keyword: str) -> Optional[Tool]:
        if keyword in self.tool_map.keys():
            used_tool = self.tool_map[keyword][0]
            if len(self.tool_map[keyword]) > 1:
                self.tool_map[keyword] = self.tool_map[keyword][1:]
            return used_tool
        else:
            return None
