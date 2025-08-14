from __future__ import annotations
from autogpt_p.helpers.singleton import Singleton
from autogpt_p.tools.tool_selector import ToolSelector


class ToolSelectorFactory(Singleton):

    _instance = None

    @classmethod
    def get_instance(cls) -> ToolSelectorFactory:
        return cls._instance

    def __init__(self, tool_selector=None):
        self.tool_selector = tool_selector

    def produce_tool_selector(self) -> ToolSelector:
        return self.tool_selector
