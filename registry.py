# registry.py
from typing import Dict, List, Any
from langchain_core.tools import StructuredTool

class PluginRegistry:
    def __init__(self):
        self.tools = {}; self.agents = {}
    def register_tool(self, tool): self.tools[tool.name] = tool
    def register_agent(self, name, system_prompt, tool_names=None, model_profile=None, llm_overrides=None, metadata=None):
        active = []
        if tool_names:
            for t in tool_names:
                if isinstance(t, dict):
                    if t.get("enabled", True): active.append(t["name"])
                elif isinstance(t, str): active.append(t)
        self.agents[name] = {"system_prompt": system_prompt, "tool_names": active, "model_profile": model_profile, "llm_overrides": llm_overrides or {}, "metadata": metadata or {}}
    def get_tools_for_agent(self, name):
        cfg = self.agents.get(name, {})
        return [self.tools[t] for t in cfg.get("tool_names", []) if t in self.tools]
