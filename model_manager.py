# model_manager.py
"""Model manager with retry logic and circuit breaker."""
import json, threading, time, logging
from typing import Dict, Optional, Any
from langchain_openai import ChatOpenAI
from resource_manager import ResourceManager
from core import ResilientLLMCaller, CircuitBreaker, LLM_RETRY, llm_logger, event_bus, EventType, logger

class SmartModelManager:
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_API_KEY = "dummy-key"
    DEFAULT_MODEL = "qwen2.5:0.5b"
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.profiles, self.current_profile_name, self.rm = {}, "default", resource_manager
        self._cache, self._cache_lock, self._prewarmed = {}, threading.Lock(), False
        self._cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        self.load_profiles({"default": {"model_name": self.DEFAULT_MODEL, "temperature": 0.0}}, "default")
    
    def load_profiles(self, profiles: Dict, default_name: str) -> None:
        if not profiles: return
        self.profiles = profiles
        self.current_profile_name = default_name if default_name in profiles else (list(profiles.keys())[0])
        with self._cache_lock: self._cache.clear(); self._prewarmed=False
        self._cb.reset()
        logger.info(f"Loaded {len(profiles)} profiles. Active: {self.current_profile_name}")
        print(f"[ModelManager] Loaded {len(profiles)} profiles. Active: {self.current_profile_name}")
    
    def set_mode(self, mode: str) -> bool:
        if mode not in self.profiles: print(f"[ModelManager] Error: '{mode}' not found"); return False
        mn = self.profiles[mode].get("model_name", "")
        if self.rm and not self.rm.can_load(mn)["can_load"]: print(f"[ModelManager] Cannot load {mn}"); return False
        old = self.current_profile_name; self.current_profile_name = mode; self._cb.reset()
        print(f"[ModelManager] Switched to {mode}")
        event_bus.publish(Event(type=EventType.MODEL_SWITCHED, data={"old":old,"new":mode}, source="ModelManager"))
        return True
    
    def _create_llm(self, pname: str, overrides: Optional[Dict] = None) -> Any:
        if pname not in self.profiles: pname = self.current_profile_name
        c = self.profiles.get(pname, {})
        opts = {"temperature": max(0.0,min(2.0,c.get("temperature",0.0))), "top_p": max(0.0,min(1.0,c.get("top_p",1.0))), "max_tokens": min(c.get("max_tokens",2048),4096)}
        if overrides:
            for k,v in overrides.items():
                if k in {"temperature","top_p","frequency_penalty","presence_penalty","max_tokens"}: opts[k] = min(v,4096) if k=="max_tokens" else v
        return ChatOpenAI(base_url=c.get("base_url", self.DEFAULT_BASE_URL), api_key=c.get("api_key", self.DEFAULT_API_KEY), model=c.get("model_name", self.DEFAULT_MODEL), **opts)
    
    def get_llm(self, override_profile: Optional[str] = None, llm_overrides: Optional[Dict] = None) -> Any:
        pname = override_profile or self.current_profile_name
        if pname not in self.profiles: pname = "default"
        key = f"{pname}:{json.dumps(llm_overrides or {}, sort_keys=True)}"
        with self._cache_lock:
            if key in self._cache: return self._cache[key]
        llm = self._create_llm(pname, llm_overrides)
        resilient = ResilientLLMCaller(llm, LLM_RETRY, self._cb)
        with self._cache_lock: self._cache[key] = resilient
        return resilient
    
    def get_raw_llm(self, override_profile=None, llm_overrides=None) -> Any:
        return self._create_llm(override_profile or self.current_profile_name, llm_overrides)
    
    def prewarm(self) -> None:
        if self._prewarmed: return
        def w(pn):
            try:
                t0=time.time(); self.get_raw_llm(pn).invoke("hi", max_tokens=1); llm_logger.log_response(f"prewarm-{pn}",2,(time.time()-t0)*1000)
            except Exception as e: llm_logger.log_error(f"prewarm-{pn}", e)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(self.profiles),4)) as p: list(p.map(w, self.profiles.keys()))
        with self._cache_lock: self._prewarmed=True
        print("[ModelManager] LLMs pre-warmed")
    
    @property
    def current_model_name(self) -> str: return self.profiles.get(self.current_profile_name, {}).get("model_name", self.DEFAULT_MODEL)
    @property
    def circuit_breaker_state(self) -> str: return self._cb.state.value