# main.py
"""Main entry point."""
import logging
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from core import logging_manager, event_bus, EventType, Event, CheckpointStorage, logger
from model_manager import SmartModelManager
from config_loader import PromptManager, PluginRegistry, ConfigLoader
from state import MemoryManager
from resource_manager import ResourceManager
from tools import get_builtin_tools
from graph_builder import build_graph
from cli import run_cli

def main():
    logging_manager
    logger.info("Application starting")
    
    rm = ResourceManager(interval=5.0)
    pm = PromptManager()
    pm.set_global("app_name", "Multi-Agent CLI")
    pm.set_global("current_date", datetime.now().strftime("%Y-%m-%d"))
    
    mm = SmartModelManager(resource_manager=rm)
    mem = MemoryManager(max_messages=50, compression_threshold=4)
    try: mem.set_summary_llm(mm.get_raw_llm())
    except: pass
    
    reg = PluginRegistry(); reg._mm = mm
    for tool in get_builtin_tools(): reg.register_tool(tool)
    
    rm.start()
    CheckpointStorage(db_path="checkpoints.db")
    
    cp = MemorySaver()
    try: app = build_graph(reg, mm, cp)
    except Exception as e: logger.error(f"Graph build failed: {e}"); app = None
    
    try:
        s = rm.status(); g = s.get('gpu')
        print(f"\n[System] GPU: {g['name'] if g else 'None'} | CPU: {s.get('cpu',{}).get('available_mb',0):.0f}MB free")
    except: print("\n[System] Resource info unavailable")
    
    print("[System] AlphaEvolve + AutoResearch enabled")
    print(f"[System] Circuit breaker: {mm.circuit_breaker_state}")
    
    event_bus.publish(Event(type=EventType.SYSTEM_STARTUP, data={"version":"2.0.0"}, source="main"))
    
    try: run_cli(reg, mm, app, mem, pm, rm, cp)
    finally:
        event_bus.publish(Event(type=EventType.SYSTEM_SHUTDOWN, source="main"))
        rm.stop()

if __name__ == "__main__":
    main()