# graph_builder.py
"""Graph builder for agent orchestration."""
import logging, time
from enum import Enum
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END

# Safely import features that only exist in newer versions of langgraph
try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    create_react_agent = None

try:
    from langgraph.types import Command
    HAS_COMMAND = True
except ImportError:
    HAS_COMMAND = False
    Command = None

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, AIMessage
from state import AgentState, EvolutionEngine, ResearchEngine
from core import tool_logger, logger

class ParallelToolNode:
    def __init__(self, tools, timeout=30, max_workers=4):
        self.tools, self.by_name, self.timeout = list(tools), {t.name:t for t in tools}, timeout
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool_exec")
    def __call__(self, state):
        msgs = state.get("messages", [])
        if not msgs or not hasattr(msgs[-1], "tool_calls") or not msgs[-1].tool_calls: return {"messages": []}
        res, futs = [], {}
        for c in msgs[-1].tool_calls:
            t = self.by_name.get(c.get("name"))
            if t: tool_logger.log_invocation(c.get("name",""), c.get("args",{})); futs[self._pool.submit(t.invoke, c.get("args",{}))] = (c, c.get("name",""), time.time())
            else: res.append(ToolMessage(content=f"Error: Unknown tool", tool_call_id=c.get("id","")))
        for f in as_completed(futs, timeout=self.timeout):
            c, tn, t0 = futs[f]
            try:
                r=f.result(timeout=max(1,self.timeout-1)); tool_logger.log_result(tn,len(str(r)),(time.time()-t0)*1000)
                res.append(ToolMessage(content=str(r) if r else "(empty)", tool_call_id=c.get("id","")))
            except Exception as e: tool_logger.log_error(tn, e); res.append(ToolMessage(content=f"Error: {e}", tool_call_id=c.get("id","")))
        return {"messages": res}

class Pattern(Enum): SUPERVISOR="supervisor"; SWARM="swarm"; PIPELINE="pipeline"; EVOLVE="evolve"; RESEARCH="research"

def build_graph(registry, mm, cp=None, pattern=None):
    agents = list(registry.agents.keys())
    if not agents: return _empty(cp)
    cfg = getattr(registry, "_workflow_cfg", {})
    p_str = pattern or (cfg.get("pattern", "supervisor") if isinstance(cfg, dict) else "supervisor")
    try: pat = Pattern(p_str)
    except: pat = Pattern.SUPERVISOR
    if pat == Pattern.EVOLVE: return _evolve(registry, mm, cp)
    if pat == Pattern.RESEARCH: return _research(registry, mm, cp)
    
    workers = {}
    with ThreadPoolExecutor(max_workers=max(1,len(agents)), thread_name_prefix="build") as pool:
        futs = {pool.submit(_mk_worker, registry, n, mm):n for n in agents}
        for f in as_completed(futs, 30):
            n=futs[f]
            try: workers[n]=f.result(timeout=30)
            except: workers[n]=_fallback(n)
    return {Pattern.SUPERVISOR:_sup, Pattern.SWARM:_swarm, Pattern.PIPELINE:_pipe}.get(pat, _sup)(registry, workers, agents, cfg if isinstance(cfg,dict) else {}, cp)

def _empty(cp):
    w=StateGraph(AgentState); w.add_node("idle",lambda s:{"messages":[]}); w.set_entry_point("idle"); w.add_edge("idle",END); return w.compile(checkpointer=cp)
def _fallback(n):
    w=StateGraph(AgentState); w.add_node(n,lambda s:{"messages":[]}); w.set_entry_point(n); w.add_edge(n,END); return w.compile()

def _mk_worker(reg, n, mm):
    c=reg.agents[n]; t=reg.get_tools_for_agent(n); l=mm.get_raw_llm(override_profile=c.get("model_profile"),llm_overrides=c.get("llm_overrides"))
    if create_react_agent:
        try: return create_react_agent(l, t, prompt=c.get("system_prompt",""))
        except Exception: pass # Fall back to manual builder if args mismatch or version issue
    return _manual(l, t, c.get("system_prompt",""))

def _manual(l, t, p):
    w=StateGraph(AgentState); ch=(ChatPromptTemplate.from_messages([("system",p),MessagesPlaceholder("messages")]) | (l.bind_tools(t) if t else l))
    def fn(s): return {"messages":[ch.invoke({"messages":s["messages"]})]}
    w.add_node("agent",fn)
    if t:
        w.add_node("tools",ParallelToolNode(t))
        w.add_conditional_edges("agent",lambda s:"tools" if s["messages"] and hasattr(s["messages"][-1],"tool_calls") and s["messages"][-1].tool_calls else END,{"tools":"tools",END:END})
        w.add_edge("tools","agent")
    else: w.add_edge("agent",END)
    w.set_entry_point("agent"); return w.compile()

def _sup(reg, w, a, c, cp):
    wg=StateGraph(AgentState)
    for n in a:
        if n in w: wg.add_node(n, w[n])
    sl = mm.get_raw_llm() if (mm:=getattr(reg,"_mm",None)) else None
    if not sl:
        from model_manager import SmartModelManager; sl = SmartModelManager._create_llm(None, "default")
    sp=ChatPromptTemplate.from_messages([("system","Pick ONE worker. Reply ONLY name.\nWorkers:\n"+"\n".join(f"- {x}" for x in a)+"\nIf done: FINISH"),MessagesPlaceholder("messages")])
    
    def sf(s):
        r=(sp|sl).invoke({"messages":s["messages"]}); ct=r.content.strip().lower()
        if HAS_COMMAND:
            if "finish" in ct: return Command(goto=END)
            for x in a:
                if x.replace("_","") in ct.replace("_","").replace(" ",""): return Command(goto=x)
            return Command(goto=a[0] if a else END)
        else:
            # Fallback for older langgraph versions (returns strings)
            if "finish" in ct: return END
            for x in a:
                if x.replace("_","") in ct.replace("_","").replace(" ",""): return x
            return a[0] if a else END

    wg.add_node("supervisor", sf)
    
    if HAS_COMMAND:
        wg.add_edge("supervisor", END)
    else:
        wg.add_conditional_edges("supervisor", sf, {x: x for x in a} | {END: END})
        
    for x in a:
        if x in w: wg.add_edge(x, "supervisor")
    wg.set_entry_point("supervisor"); return wg.compile(checkpointer=cp)

def _swarm(reg, w, a, c, cp):
    wg=StateGraph(AgentState)
    for n in a:
        if n in w: wg.add_node(n, w[n])
    
    def router_fn(s):
        if HAS_COMMAND:
            return Command(goto=a[0]) if a else Command(goto=END)
        else:
            return a[0] if a else END
            
    wg.add_node("router", router_fn)
    if a: wg.add_edge("router", a[0])
    
    def mk(cur):
        def r(s):
            if s.get("messages"):
                ct=str(s["messages"][-1].content).lower()
                for x in a:
                    if x!=cur and x.replace("_","") in ct.replace("_","").replace(" ",""):
                        if HAS_COMMAND: return Command(goto=x)
                        else: return x
            if HAS_COMMAND: return Command(goto=END)
            else: return END
        return r

    for x in a:
        opts = {y: y for y in a if y != x} | {END: END}
        wg.add_conditional_edges(x, mk(x), opts)
        
    wg.set_entry_point("router"); return wg.compile(checkpointer=cp)

def _pipe(reg, w, a, c, cp):
    wg=StateGraph(AgentState); st=c.get("stages",a); v=[s for s in st if s in w] or [x for x in a if x in w][:1]
    for n in v: wg.add_node(n,w[n])
    for i in range(len(v)-1): wg.add_edge(v[i],v[i+1])
    if v: wg.add_edge(v[-1],END); wg.set_entry_point(v[0])
    else: wg.add_node("e",lambda s:{"messages":[]}); wg.add_edge("e",END); wg.set_entry_point("e")
    return wg.compile(checkpointer=cp)

def _infer_scaffold(task):
    tl=task.lower()
    if "prime" in tl: return "",[{"input":(2,),"expected":True},{"input":(4,),"expected":False},{"input":(17,),"expected":True}]
    if "sort" in tl: return "",[{"input":([3,1,2],),"expected":[1,2,3]}]
    if "fib" in tl: return "",[{"input":(0,),"expected":0},{"input":(5,),"expected":5}]
    return "",[{"input":(2,),"expected":True},{"input":(4,),"expected":False}]

def _evolve(reg, mm, cp):
    wg=StateGraph(AgentState)
    def ev(s):
        m=s.get("messages",[]); t=m[-1].content if m else "Write fn"
        mgr=getattr(reg,"_mm",None)
        if not mgr: return {"messages":[AIMessage(content="[Error] No MM")]}
        ec=getattr(reg,"_evo_cfg",{}); sc,tc=ec.get("scaffold",""),ec.get("test_cases",[])
        if not sc and not tc: sc,tc=_infer_scaffold(t)
        try:
            eng=EvolutionEngine(t,sc,tc,mgr.get_raw_llm(),ec.get("num_islands",2),ec.get("pop_per_island",3),ec.get("max_generations",2),ec.get("eval_timeout",3.0))
            b=eng.run()
            for l in eng.log: print(f"  {l}")
            return {"messages":[AIMessage(content=f"[AlphaEvolve Complete]\nIslands: {eng.num_islands} | Evals: {eng._total_evals}\nArchive: {len(eng.archive.entries)}\nFitness: {b.fitness:.2f}\n\n```python\n{b.code}\n```")]}
        except Exception as e: return {"messages":[AIMessage(content=f"[Error] {e}")]}
    wg.add_node("evolve",ev); wg.set_entry_point("evolve"); wg.add_edge("evolve",END); return wg.compile(checkpointer=cp)

def _research(reg, mm, cp):
    wg=StateGraph(AgentState)
    def rs(s):
        m=s.get("messages",[]); t=m[-1].content if m else "Research"
        mgr=getattr(reg,"_mm",None)
        if not mgr: return {"messages":[AIMessage(content="[Error] No MM")]}
        rc=getattr(reg,"_research_cfg",{})
        try:
            eng=ResearchEngine(t,mgr.get_raw_llm(),rc.get("critique_rounds",1)); rep=eng.run()
            for l in eng.log: print(f"  {l}")
            return {"messages":[AIMessage(content=f"[Research Complete]\nPhases: {len(eng.findings)} | Confidence: {eng.confidence:.0f}/10\n\n--- Report ---\n{rep}")]}
        except Exception as e: return {"messages":[AIMessage(content=f"[Error] {e}")]}
    wg.add_node("research",rs); wg.set_entry_point("research"); wg.add_edge("research",END); return wg.compile(checkpointer=cp)