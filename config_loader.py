# config_loader.py
"""Configuration loader, prompt manager, and plugin registry."""
import yaml, re, logging
from typing import Dict, List, Optional, Any
from langchain_core.tools import StructuredTool
from core import validate_config, event_bus, EventType, logger

class PromptTemplate:
    def __init__(self, name: str, content: str):
        self.name, self.variables, self.versions, self.current = name, re.findall(r'\{\{(\w+)\}\}', content), {"1.0": content}, "1.0"
    def add_version(self, v, c): self.versions[v] = c
    def render(self, vars=None, version=None):
        c = self.versions.get(version or self.current, "")
        if vars:
            for k, v in (vars or {}).items(): c = c.replace(f"{{{{{k}}}}}", str(v))
        return c

class PromptManager:
    def __init__(self): self.templates, self.globals = {}, {}
    def register(self, name, content): pt=PromptTemplate(name,content); self.templates[name]=pt; return pt
    def render(self, name, vars=None, version=None):
        pt=self.templates.get(name)
        if not pt: return f"[Missing: {name}]"
        return pt.render({**self.globals, **(vars or {})}, version)
    def set_global(self, k, v): self.globals[k] = v

class PluginRegistry:
    def __init__(self):
        self.tools, self.agents = {}, {}
        self._workflow_cfg = self._evo_cfg = self._research_cfg = None
        self._mm = None
    def register_tool(self, tool):
        if not hasattr(tool,'name') or not tool.name: raise ValueError("Tool needs name")
        self.tools[tool.name] = tool
    def register_agent(self, name, system_prompt, tool_names=None, model_profile=None, llm_overrides=None, metadata=None):
        active = []
        if tool_names:
            for t in tool_names:
                if isinstance(t, dict):
                    if t.get("enabled", True) and t.get("name"): active.append(t["name"])
                elif isinstance(t, str) and t: active.append(t)
        self.agents[name] = {"system_prompt": system_prompt or "", "tool_names": active, "model_profile": model_profile, "llm_overrides": llm_overrides or {}, "metadata": metadata or {}}
    def get_tools_for_agent(self, name):
        c = self.agents.get(name, {})
        return [self.tools[t] for t in c.get("tool_names", []) if t in self.tools]

class ConfigLoader:
    @staticmethod
    def load(filepath, registry, model_manager, prompt_manager=None):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = yaml.safe_load(f)
        except Exception as e: print(f"[Error] Cannot read {filepath}: {e}"); return False
        if not data: print("[Error] Empty config"); return False
        
        errs = validate_config(data)
        if errs:
            for err in errs[:3]: print(f"[Warning] Config: {err.path} - {err.message}")
            event_bus.publish(Event(type=EventType.CONFIG_ERROR, data={"errors": len(errs)}, source="ConfigLoader"))
        
        errors = []
        if "model_profiles" in data:
            try: model_manager.load_profiles(data["model_profiles"], data.get("defaults", {}).get("model_profile", "default"))
            except Exception as e: errors.append(f"Profiles: {e}")
        
        for tc in data.get("tools", []):
            try:
                t = ConfigLoader._make_tool(tc)
                if t: registry.register_tool(t)
            except Exception as e: errors.append(f"Tool: {e}")
        
        pm = prompt_manager
        if pm and "prompt_templates" in data:
            for n, v in data["prompt_templates"].items():
                try:
                    if isinstance(v, str): pm.register(n, v)
                    elif isinstance(v, dict):
                        pt = pm.register(n, v.get("content",""))
                        for ver, vd in v.get("versions",{}).items(): pt.add_version(ver, vd.get("content","") if isinstance(vd,dict) else vd)
                except Exception as e: errors.append(f"Template: {e}")
        if pm and "prompt_globals" in data: pm.globals.update(data["prompt_globals"])
        
        if "agents" in data:
            for a in data["agents"]:
                try:
                    blocks = a.get("prompt_blocks", [])
                    fp = "\n".join(pm.render(b[1:]) if b.startswith("$") else b for b in blocks)
                    registry.register_agent(name=a["name"], system_prompt=fp, tool_names=a.get("tools",[]), model_profile=a.get("model_profile"), llm_overrides=a.get("llm_overrides"), metadata=a.get("metadata"))
                except Exception as e: errors.append(f"Agent: {e}")
        
        if "evolution" in data: registry._evo_cfg = data["evolution"]
        if "research" in data: registry._research_cfg = data["research"]
        if "workflow" in data: registry._workflow_cfg = data["workflow"]
        
        print(f"[System] Configuration loaded from {filepath}")
        if errors:
            for err in errors: print(f"[Warning] {err}")
        event_bus.publish(Event(type=EventType.CONFIG_LOADED, data={"file":filepath}, source="ConfigLoader"))
        return True

    @staticmethod
    def _make_tool(conf):
        t = conf.get("type"); name = conf.get("name")
        if not name: return None
        if t == "python":
            scope = {}
            try:
                exec(conf.get("code","",{}), {}, scope); fn=scope.get("run_tool")
                if fn: return StructuredTool.from_function(func=fn, name=name, description=conf.get("description",""))
            except Exception as e: print(f"[Error] Tool '{name}' compile: {e}")
        elif t == "duckduckgo":
            try:
                from langchain_community.tools import DuckDuckGoSearchRun; return DuckDuckGoSearchRun(name=name)
            except ImportError: print("[Error] duckduckgo-search not installed")
        return None