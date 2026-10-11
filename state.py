# state.py
"""State definitions and engines for AlphaEvolve and AutoResearch."""
import threading, time, hashlib, ast, random, logging, re, io
from typing import TypedDict, Annotated, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages
from concurrent.futures import ThreadPoolExecutor, as_completed
from core import CodeSandbox, event_bus, EventType, Event, logger

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class EvalResult:
    __slots__ = ("passed","failed","errors","score","runtime_ms")
    def __init__(self, p=0, f=0, e=0, s=0.0, t=0.0): self.passed,self.failed,self.errors,self.score,self.runtime_ms = p,f,e,s,t

class Solution:
    def __init__(self, code, island=0, gen=0, pids=None):
        self.code,self.island,self.gen,self.parent_ids = code,island,gen,pids or []
        self.signature = hashlib.sha256(code.encode()).hexdigest()[:12]; self.eval_result = None
    @property
    def fitness(self): return self.eval_result.score if self.eval_result else 0.0

class Archive:
    def __init__(self, max_sz=20): self.entries, self.max_sz, self._lock = [], max_sz, threading.Lock()
    def try_add(self, sol):
        with self._lock:
            if self.entries and sol.fitness < min(e.solution.fitness for e in self.entries) + 0.01: return False
            if any(e.solution.signature==sol.signature for e in self.entries): return False
            self.entries.append(type('E',(object,),{'solution':sol,'added_at':time.time()})())
            if len(self.entries)>self.max_sz: self.entries.sort(key=lambda e:e.solution.fitness,reverse=True); self.entries=self.entries[:self.max_sz]
            return True
    def sample(self, c=2):
        with self._lock: return [e.solution for e in random.sample(self.entries,min(c,len(self.entries)))] if self.entries else []

class Island:
    def __init__(self, iid, sc, tests, llm, to=5.0, arch=None):
        self.id,self.scaffold,self.test_cases,self.llm,self.archive = iid,sc,tests,llm,arch
        self.population,self.generation,self.best,self.total_evals,self.log = [],0,None,0,[]
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"island_{iid}")

class EvolutionEngine:
    def __init__(self, task, sc, tests, llm, ni=2, pop=3, mg=3, to=5.0, ma=16):
        self.task,self.scaffold,self.tests,self.llm = task,sc,tests,llm
        self.num_islands,self.pop,self.max_gen,self.timeout = ni,pop,mg,to
        self.archive,self.islands,self.best_overall,self.log = Archive(ma),[],None,[]
        self._total_evals = 0; self._sandbox = CodeSandbox(timeout=to)
    
    def _exec(self, code):
        t0=time.time(); p,f,e=0,0,0
        for tc in self.test_cases:
            inp,exp=tc.get("input",()),tc.get("expected")
            sb=f"{self.scaffold}\n{code}\n_args={repr(inp)}\n_expected={repr(exp)}\ntry:\n  _r=solve(*_args)\n  print('PASS' if _r==_expected else 'FAIL')\nexcept:\n  print('ERROR')\n"
            r=self._sandbox.execute(sb)
            if r.success:
                if "PASS" in r.output: p+=1
                elif "ERROR" in r.output: e+=1
                else: f+=1
            else: e+=1
        tot=p+f+e; ms=(time.time()-t0)*1000
        return EvalResult(p,f,e,(p/tot if tot else 0)+(0.05 if ms<self.timeout*1000*len(self.tests) else 0),ms)
    
    def _gen(self, prompt):
        try:
            r=self.llm.invoke(prompt,max_tokens=512); c=r.content
            try:
                tree=ast.parse(c)
                for n in ast.walk(tree):
                    if isinstance(n,ast.FunctionDef) and n.name=="solve": return ast.unparse(n)
            except: pass
            m=re.search(r'(def\s+solve\s*\([^)]*\)\s*:\n(?:\s{4,}.+\n)*)',c,re.M)
            return m.group(1).strip() if m else "def solve(*a): return None"
        except: return "def solve(*a): return None"
    
    def _mutate(self, s, i):
        h=""
        if self.archive.entries: sa=self.archive.sample(1); h=f"\nDifferent:\n```python\n{sa[0].code}\n```" if sa else ""
        return Solution(self._gen(f"Task: {self.task}\nCur ({s.eval_result.passed if s.eval_result else 0}/{len(self.tests)}):\n```python\n{s.code}\n```{h}\nImprove. ONLY def solve(...)."), i.id, i.generation+1, [s.signature])
    
    def _seed(self, isl):
        sols=[Solution(self._gen(f"Task: {self.task}\nWrite def solve(...). Output ONLY function."),isl.id,0) for _ in range(self.pop)]
        futs={isl._pool.submit(self._exec,s.code):s for s in sols}
        for fu in as_completed(futs,timeout=self.timeout*len(self.tests)+10):
            s=futs[fu]
            try: s.eval_result=fu.result(timeout=30)
            except: s.eval_result=EvalResult(errors=len(self.tests))
            isl.total_evals+=1; self._total_evals+=1; self.archive.try_add(s)
        isl.population=sorted(sols,key=lambda x:x.fitness,reverse=True); isl.best=isl.population[0] if isl.population else None
    
    def _evolve_isl(self, isl):
        isl.generation+=1; new=[]
        if isl.best: el=Solution(isl.best.code,isl.id,isl.generation,[isl.best.signature]); el.eval_result=isl.best.eval_result; new.append(el)
        while len(new)<self.pop:
            if random.random()<0.2 and len(isl.population)>=2: new.append(Solution(self._gen(f"Task: {self.task}\nA:\n```python\n{self._tournament(isl.population).code}\n```\nB:\n```python\n{self._tournament(isl.population).code}\n```\nCombine."),isl.id,isl.generation+1))
            elif random.random()<0.3 and self.archive.entries: sa=self.archive.sample(1); new.append(Solution(self._gen(f"Task: {self.task}\nAdapt:\n```python\n{sa[0].code}\n```"),isl.id,isl.generation+1))
            else: new.append(self._mutate(self._tournament(isl.population) if isl.population else Solution("def solve(*a): return None"),isl))
        futs={isl._pool.submit(self._exec,s.code):s for s in new if s.eval_result is None}
        for fu in as_completed(futs,timeout=self.timeout*len(self.tests)+10):
            s=futs[fu]
            try: s.eval_result=fu.result(timeout=30)
            except: s.eval_result=EvalResult(errors=len(self.tests))
            isl.total_evals+=1; self._total_evals+=1; self.archive.try_add(s)
        isl.population=sorted(new,key=lambda x:x.fitness,reverse=True); isl.best=isl.population[0] if isl.population else None
        if isl.best and (not self.best_overall or isl.best.fitness>self.best_overall.fitness): self.best_overall=isl.best
    
    def _tournament(self, p, k=2): return max(random.sample(p,min(k,len(p))),key=lambda x:x.fitness)
    
    def run(self):
        self.log=[f"[AlphaEvolve] {self.num_islands} islands x {self.pop} pop x {self.max_gen} gen"]
        if not self.tests: self.log.append("[AlphaEvolve] Warning: No tests"); return Solution("def solve(*a): return None")
        event_bus.publish(Event(type=EventType.EVOLVE_START, data={"task":self.task[:100]}, source="EvolutionEngine"))
        self.islands=[Island(i,self.scaffold,self.tests,self.llm,self.timeout,self.archive) for i in range(self.num_islands)]
        with ThreadPoolExecutor(max_workers=self.num_islands) as p:
            futs={p.submit(self._seed,i):i for i in self.islands}
            for f in as_completed(futs,120):
                try: f.result(timeout=120)
                except: pass
        for _ in range(self.max_gen):
            self.log.append(f"[AlphaEvolve] Gen {_+1}/{self.max_gen}")
            with ThreadPoolExecutor(max_workers=self.num_islands) as p:
                futs={p.submit(self._evolve_isl,i):i for i in self.islands}
                for f in as_completed(futs,180):
                    try: f.result(timeout=180)
                    except: pass
        for i in self.islands: self.log.extend(i.log); i._pool.shutdown(wait=False)
        self.log.append(f"[AlphaEvolve] Done. {self._total_evals} evals.")
        event_bus.publish(Event(type=EventType.EVOLVE_COMPLETE, source="EvolutionEngine"))
        return self.best_overall or Solution("def solve(*a): return None")

class ResearchEngine:
    def __init__(self, topic, llm, max_rounds=2):
        self.topic,self.llm,self.max_rounds,self.confidence,self.log = topic,llm,max_rounds,0.0,[]
        self.findings, self.critique_count = {}, 0
    def _phase(self, ph, ctx=""):
        ps={"hypothesis":f"Topic: {self.topic}\nGenerate 2-3 hypotheses. H1:...","experiment_design":f"Topic: {self.topic}\nHypotheses:\n{ctx}\nDesign experiment.","execute":f"Topic: {self.topic}\nPlan:\n{ctx}\nSimulate.","analyze":f"Topic: {self.topic}\nData:\n{ctx}\nAnalyze.","synthesize":f"Topic: {self.topic}\nFindings:\n{ctx}\nSynthesize.","critique":f"Topic: {self.topic}\nSynthesis:\n{ctx}\nCritique. Rate 0-10.","report":f"Topic: {self.topic}\nFindings:\n{ctx}\nWrite: ## Abstract ## Methods ## Results ## Conclusion"}
        try: return self.llm.invoke(ps.get(ph,""),max_tokens=1024).content
        except Exception as e: return f"[Error: {e}]"
    def run(self):
        event_bus.publish(Event(type=EventType.RESEARCH_START, source="ResearchEngine"))
        self.log=[f"[Research] Topic: {self.topic[:80]}"]
        for ph in ["hypothesis","experiment_design","execute","analyze","synthesize","critique"]:
            ctx="\n---\n".join(f"{k}: {v}" for k,v in self.findings.items() if v)
            self.findings[ph]=self._phase(ph,ctx); self.log.append(f"[Research] {ph}")
            if ph=="critique":
                self.critique_count+=1
                try:
                    for l in self.findings[ph].split("\n"):
                        if "confidence" in l.lower() or "rating" in l.lower():
                            d="".join(c for c in l if c.isdigit() or c==".")
                            if d: self.confidence=max(0,min(10,float(d[:3]))); break
                except: self.confidence=5.0
                if self.confidence<6.0 and self.critique_count<self.max_rounds: self.log.append(f"[Research] Revise"); continue
        ctx="\n---\n".join(f"{k}: {v}" for k,v in self.findings.items() if v)
        rep=self._phase("report",ctx); self.findings["final_report"]=rep
        self.log.append(f"[Research] Done. Confidence: {self.confidence:.0f}/10")
        event_bus.publish(Event(type=EventType.RESEARCH_COMPLETE, source="ResearchEngine"))
        return rep

class MemoryManager:
    CHR, CW = 0.25, {"qwen2.5:0.5b":32768,"qwen2.5:7b":32768,"qwen3:8b":32768,"llama3:8b":8192}
    def __init__(self, max_msgs=50, comp_thresh=4): self.max_msgs,self.comp_thresh,self._llm = max_msgs,comp_thresh,None
    def set_summary_llm(self, l): self._llm=l
    def est_tok(self, m): return int(sum(len(str(x.content)) for x in m)*self.CHR)
    def compress(self, m, l=None):
        if len(m)<=self.comp_thresh: return m, False
        old,rec = m[:-self.comp_thresh], m[-self.comp_thresh:]
        s=self._sum(old, l or self._llm)
        if s: event_bus.publish(Event(type=EventType.MEMORY_COMPRESSED, data={"from":len(m),"to":len(rec)+1}, source="MemoryManager")); return [SystemMessage(content=f"[Summary]\n{s}")]+rec, True
        return m, False
    def _sum(self, m, l):
        c="\n".join(f"{'User' if isinstance(x,HumanMessage) else 'Asst'}: {str(x.content)[:120]}" for x in m)
        if l:
            try: return l.invoke(f"Summarize:\n{c}\nSummary:",max_tokens=256).content
            except: pass
        return "Earlier:\n"+"\n".join(f"- {str(x.content)[:80]}" for x in m[:8])
    def stats(self, m, mn):
        t=self.CW.get(mn,4096); u=self.est_tok(m); p=u/t if t else 0
        if p>0.8: event_bus.publish(Event(type=EventType.MEMORY_WARNING, data={"pct":round(p*100,1)}, source="MemoryManager"))
        return {"model":mn,"total_tokens":t,"used_tokens":u,"message_count":len(m),"warning":p>0.8}