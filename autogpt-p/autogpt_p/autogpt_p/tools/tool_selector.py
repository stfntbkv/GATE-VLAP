from typing import List, Optional

from autogpt_p.tools.tool import Tool


class ToolSelector:

    def __init__(self, tools: List[Tool]):
        self.tool_map = {}
        for tool in tools:
            if tool.keyword not in self.tool_map.keys():
                self.tool_map[tool.keyword] = [tool]
            else:
                self.tool_map[tool.keyword].append(tool)

    def get_tool(self, keyword: str) -> Optional[Tool]:
        if keyword in self.tool_map.keys():
            return self.tool_map[keyword][0]
        else:
            return None

    def get_tool_in(self, string_with_keyword: str) -> Optional[Tool]:
        longest_match = ""
        for keyword in self.tool_map.keys():
            if keyword in string_with_keyword and len(keyword) > len(longest_match):
                longest_match = keyword
        return self.tool_map[longest_match][0] if len(longest_match) > 0 else None

    def get_all_tools(self) -> List[Tool]:
        return [v[0] for k, v in self.tool_map.items()]


def from_classes(memory, tool_classes) -> ToolSelector:
    return ToolSelector([tool(memory) for tool in tool_classes])
