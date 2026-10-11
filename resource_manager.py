# resource_manager.py
"""Resource monitoring for model selection."""
import subprocess, threading, time, logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from core import logger

@dataclass
class ResourceSnapshot:
    timestamp: float = 0.0; gpu_name: str = ""; gpu_mem_used: float = 0; gpu_mem_total: float = 0
    gpu_util: float = 0; cpu_mem_used: float = 0; cpu_mem_total: float = 0; cpu_percent: float = 0
    @property
    def gpu_available(self) -> float: return max(0, self.gpu_mem_total - self.gpu_mem_used)
    @property
    def cpu_available(self) -> float: return max(0, self.cpu_mem_total - self.cpu_mem_used)
    def to_dict(self) -> Dict:
        r = {"has_gpu": bool(self.gpu_mem_total), "gpu": {"name": self.gpu_name, "mem_used_mb": round(self.gpu_mem_used), "mem_total_mb": round(self.gpu_mem_total), "available_mb": round(self.gpu_available)} if self.gpu_mem_total else None,
             "cpu": {"mem_used_mb": round(self.cpu_mem_used), "mem_total_mb": round(self.cpu_mem_total), "available_mb": round(self.cpu_available)}}
        return r

@dataclass
class ModelSize:
    name: str; gpu_mb: float; cpu_mb: float; can_cpu: bool = True

MODEL_SIZES = {
    "qwen2.5:0.5b": ModelSize("qwen2.5:0.5b", 800, 500), "qwen2.5:1.5b": ModelSize("qwen2.5:1.5b", 1500, 1000),
    "qwen2.5:3b": ModelSize("qwen2.5:3b", 2500, 2000), "qwen2.5:7b": ModelSize("qwen2.5:7b", 5500, 4500, False),
    "qwen3:0.6b": ModelSize("qwen3:0.6b", 900, 550), "qwen3:8b": ModelSize("qwen3:8b", 6000, 5000, False),
}

class ResourceManager:
    def __init__(self, interval=5.0):
        self.interval, self.snapshot, self.history = interval, ResourceSnapshot(), []
        self._thread, self._running, self.has_gpu, self._gpu_checked, self._lock = None, False, False, False, threading.Lock()
    
    def _detect_gpu(self) -> bool:
        try: r=subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True,timeout=2); return r.returncode==0 and bool(r.stdout.strip())
        except: return False
    def _gpu_info(self) -> Dict:
        if not self.has_gpu: return {}
        try:
            r=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu,name","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=2)
            if r.returncode==0: p=r.stdout.strip().split(", "); return {"used":float(p[0]),"total":float(p[1]),"util":float(p[2]),"name":p[3] if len(p)>3 else "GPU"}
        except: pass
        return {}
    def _cpu_info(self) -> Dict:
        try:
            import psutil; m=psutil.virtual_memory(); return {"used":m.used/1048576,"total":m.total/1048576,"pct":m.percent}
        except ImportError: pass
        try:
            r=subprocess.run(["free","-m"],capture_output=True,text=True,timeout=2)
            if r.returncode==0: p=r.stdout.strip().split("\n")[1].split(); return {"used":float(p[2]),"total":float(p[1]),"pct":float(p[2])/float(p[1])*100}
        except: pass
        return {}
    
    def take_snapshot(self) -> ResourceSnapshot:
        if not self._gpu_checked: self.has_gpu=self._detect_gpu(); self._gpu_checked=True
        g,c = {},{}
        with ThreadPoolExecutor(max_workers=2) as p:
            fg,fc = p.submit(self._gpu_info), p.submit(self._cpu_info)
            try: g=fg.result(timeout=3)
            except: pass
            try: c=fc.result(timeout=3)
            except: pass
        s=ResourceSnapshot(time.time(),g.get("name",""),g.get("used",0),g.get("total",0),g.get("util",0),c.get("used",0),c.get("total",0),c.get("pct",0))
        with self._lock: self.snapshot=s; self.history.append(s)
        if len(self.history)>500: self.history=self.history[-500:]
        return s
    def _loop(self):
        while self._running: self.take_snapshot(); time.sleep(self.interval)
    def start(self):
        if not self._running: self._running=True; self._thread=threading.Thread(target=self._loop,daemon=True); self._thread.start(); self.take_snapshot()
    def stop(self): self._running=False
    def get_size(self, model): return MODEL_SIZES.get(model) or next((v for k,v in MODEL_SIZES.items() if k.split(":")[0]==model.split(":")[0]), None)
    def can_load(self, model):
        sz=self.get_size(model); r={"model":model,"can_load":False,"mode":"cpu","warnings":[]}
        if not sz: r["can_load"]=True; return r
        with self._lock: snap=self.snapshot
        if self.has_gpu and snap.gpu_available>=sz.gpu_mb: r["can_load"],r["mode"]=True,"gpu"
        elif sz.can_cpu and snap.cpu_available>=sz.cpu_mb: r["can_load"]=True
        else: r["warnings"].append(f"Need GPU:{sz.gpu_mb} CPU:{sz.cpu_mb}MB")
        return r
    def suggest_model(self):
        with self._lock: snap=self.snapshot
        for n in ["qwen3:8b","qwen3:4b","qwen2.5:3b","qwen2.5:1.5b","qwen2.5:0.5b","qwen3:0.6b"]:
            sz=self.get_size(n)
            if sz and ((self.has_gpu and snap.gpu_available>=sz.gpu_mb) or (sz.can_cpu and snap.cpu_available>=sz.cpu_mb)): return n
        return "qwen2.5:0.5b"
    def status(self):
        with self._lock: return self.snapshot.to_dict()