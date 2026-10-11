# cli.py
"""CLI with input validation and improved UX."""
import sys, time, threading
from queue import Queue, Empty
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core import InputValidator, event_bus

V = InputValidator()
RULES = {
    "filepath": [V.not_empty, V.no_path_traversal, V.safe_filename, V.max_length(255)],
    "profile": [V.not_empty, V.max_length(50)],
    "task": [V.not_empty, V.max_length(1000)],
    "message": [V.max_length(10000)]
}

def print_help():
    print("\n--- Commands ---")
    print("  <msg>         : Chat with agents")
    print("  /load <f>     : Load YAML config")
    print("  /mode <p>     : Switch model profile")
    print("  /profiles     : List available profiles")
    print("  /list         : List agents & tools")
    print("  /resources    : Show GPU/CPU status")
    print("  /suggest      : Suggest best model")
    print("  /auto         : Auto-select model")
    print("  /memory       : Show memory stats")
    print("  /compress     : Compress conversation")
    print("  /workflow     : Show workflow config")
    print("  /evolve <t>   : Run AlphaEvolve")
    print("  /research <t> : Run AutoResearch")
    print("  /logs         : Show recent events")
    print("  /help         : Show this help")
    print("  /exit         : Quit")
    print("----------------")

def _stream(app, inp, cfg, q):
    try:
        for ev in app.stream({"messages":[HumanMessage(content=inp)]}, cfg, stream_mode="updates"):
            for n, u in ev.items():
                if "messages" not in u: continue
                for m in u["messages"]:
                    if isinstance(m, ToolMessage): q.put(("tool", str(m.content)[:300] if m.content else "(empty)"))
                    elif isinstance(m, AIMessage) and m.content: q.put(("agent", n, m.content))
    except Exception as e: q.put(("error", str(e)))
    finally: q.put(("done",))

def _run_special(app, arg, label, tout):
    print(f"\n[{label}] Starting: {arg[:70]}...")
    cfg = {"configurable": {"thread_id": f"{label}-{time.time()}"}}
    q = Queue(); threading.Thread(target=_stream, args=(app, arg, cfg, q), daemon=True).start()
    while True:
        try:
            i = q.get(timeout=tout)
            if i[0]=="done": break
            elif i[0]=="agent": print(f"\n{i[2]}")
            elif i[0]=="tool": print(f"\n[Tool] {i[1]}")
            elif i[0]=="error": print(f"\n[Error] {i[1]}")
        except Empty: print(f"\n[Error] {label} timed out"); break

def run_cli(reg, mm, app, mem, pm, rm, cp):
    config = {"configurable": {"thread_id": "default-session"}}
    print("\n" + "=" * 60 + "\n  Multi-Agent CLI — AlphaEvolve + AutoResearch\n" + "=" * 60)
    print("Ready. Type '/load agents.yaml' to start."); print_help()
    threading.Thread(target=mm.prewarm, daemon=True).start()

    def rebuild():
        nonlocal app, config
        try:
            from graph_builder import build_graph; reg._mm = mm; app = build_graph(reg, mm, cp)
            config = {"configurable": {"thread_id": f"session-{time.time()}"}}
            return True
        except Exception as e: print(f"[Error] Rebuild failed: {e}"); return False

    while True:
        sys.stdout.write("\nUser >> "); sys.stdout.flush()
        try: ui = sys.stdin.readline()
        except KeyboardInterrupt: print("\n"); break
        except EOFError: break
        if not ui: break
        ui = ui.rstrip("\n")
        if not ui.strip(): continue

        if ui.startswith("/"):
            p = ui.split(maxsplit=1); cmd, arg = p[0].lower(), p[1] if len(p)>1 else ""
            if cmd=="/exit": print("Goodbye!"); break
            elif cmd=="/help": print_help()
            elif cmd=="/load" and arg:
                errs = V.validate({"filepath":arg}, RULES)
                if errs: print(f"[Error] {errs[0]}"); continue
                from config_loader import ConfigLoader
                if ConfigLoader.load(arg, reg, mm, pm): rebuild()
            elif cmd=="/mode" and arg:
                if mm.set_mode(arg): rebuild()
            elif cmd=="/evolve" and arg:
                errs = V.validate({"task":arg}, RULES)
                if errs: print(f"[Error] {errs[0]}"); continue
                from graph_builder import build_graph; reg._mm=mm; _run_special(build_graph(reg,mm,cp,pattern="evolve"),arg,"AlphaEvolve",180)
            elif cmd=="/research" and arg:
                errs = V.validate({"task":arg}, RULES)
                if errs: print(f"[Error] {errs[0]}"); continue
                from graph_builder import build_graph; reg._mm=mm; _run_special(build_graph(reg,mm,cp,pattern="research"),arg,"AutoResearch",300)
            elif cmd=="/profiles":
                print("\n--- Profiles ---")
                for k, v in mm.profiles.items(): print(f"  {k}: {v.get('model_name')} temp={v.get('temperature',0)}{' <--' if k==mm.current_profile_name else ''}")
            elif cmd=="/list":
                print("\n--- Agents ---")
                for k, v in reg.agents.items(): print(f"  {k}: {v.get('tool_names',[])}")
                print("--- Tools ---")
                for k, v in reg.tools.items(): print(f"  {k}: {v.description[:50] if v.description else ''}")
            elif cmd=="/resources":
                s=rm.status(); print(f"\n  GPU: {'Yes' if s.get('has_gpu') else 'No'}")
                g=s.get('gpu')
                if g: print(f"  {g['name']}: {g['mem_used_mb']}/{g['mem_total_mb']}MB")
                c=s.get('cpu',{}); print(f"  CPU: {c.get('mem_used_mb',0)}/{c.get('mem_total_mb',0)}MB")
            elif cmd=="/suggest": print(f"\n  Suggested: {rm.suggest_model()}")
            elif cmd=="/auto": print(f"\n  Auto: {mm.auto_select()}"); rebuild()
            elif cmd=="/memory":
                try: st=app.get_state(config); msgs=st.values.get("messages",[]) if st.values else []
                except: msgs=[]
                if msgs:
                    stats=mem.stats(msgs,mm.current_model_name); print(f"\n  Msgs: {stats['message_count']} | Tok: {stats['used_tokens']}/{stats['total_tokens']}")
                    if stats.get("warning"): print("  Warning: high usage")
                else: print("\n  No messages.")
            elif cmd=="/compress":
                try: st=app.get_state(config); msgs=st.values.get("messages",[]) if st.values else []
                except: msgs=[]
                if msgs:
                    old=len(msgs); _,ok=mem.compress(msgs,mm.get_raw_llm())
                    if ok: config["configurable"]["thread_id"]=f"comp-{time.time()}"; print(f"\n  Compressed: {old}")
                    else: print("\n  Nothing to compress")
                else: print("\n  Nothing to compress")
            elif cmd=="/workflow":
                wc=getattr(reg,"_workflow_cfg",None); print(f"\n  Pattern: {wc.get('pattern','supervisor') if isinstance(wc,dict) else 'supervisor'}")
                ec=getattr(reg,"_evo_cfg",None)
                if ec: print(f"  Evolution: {ec.get('num_islands',2)} islands, {ec.get('max_generations',2)} gen")
                rc=getattr(reg,"_research_cfg",None)
                if rc: print(f"  Research: {rc.get('critique_rounds',1)} critique rounds")
            elif cmd=="/logs":
                evts=event_bus.get_history(limit=10); print("\n--- Recent Events ---")
                for e in evts: print(f"  [{e.type.value}] {e.data}")
            else: print(f"  Unknown: {cmd}" if cmd not in ("/load","/mode","/evolve","/research") else f"  Usage: {cmd} <arg>")
            continue

        errs = V.validate({"message":ui}, RULES)
        if errs: print(f"[Error] {errs[0]}"); continue

        q = Queue(); threading.Thread(target=_stream, args=(app, ui, config, q), daemon=True).start()
        while True:
            try:
                i = q.get(timeout=120)
                if i[0]=="done": break
                elif i[0]=="tool": print(f"\n[Tool] {i[1]}")
                elif i[0]=="agent": print(f"\n[{i[1]}]\n  {i[2]}")
                elif i[0]=="error": print(f"\n[Error] {i[1]}")
            except Empty: print("\n[Error] Timeout"); break