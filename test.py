#!/usr/bin/env python3
# test.py
import subprocess, sys, os, time, threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import List

@dataclass
class Result:
    name: str
    passed: bool
    desc: str
    detail: str = ""

class Tester:
    def __init__(self):
        self.q = Queue()
        self.lines = []
        self.proc = None
        self._mark = 0
        self._buf = ""
        
    def _reader(self):
        try:
            while True:
                c = self.proc.stdout.read(1)
                if not c:
                    if self._buf:
                        self.lines.append(self._buf)
                        self.q.put(self._buf)
                    break
                if c == "\n":
                    self.lines.append(self._buf)
                    self.q.put(self._buf)
                    self._buf = ""
                elif c == "\r":
                    continue
                else:
                    self._buf += c
        except Exception:
            if self._buf:
                self.lines.append(self._buf)
                self.q.put(self._buf)

    def wait_for(self, pattern: str, timeout: float = 30) -> tuple:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self.q.get(timeout=0.2)
                if pattern.lower() in line.lower():
                    return True, line
            except Empty:
                pass
            if pattern.lower() in self._buf.lower():
                return True, self._buf
        return False, None

    def wait_prompt(self, timeout: float = 30) -> tuple:
        return self.wait_for("User >>", timeout)

    def send(self, cmd: str, wait: bool = True, timeout: float = 180) -> tuple:
        print(f"\n  [CMD] >>> {cmd if cmd else '(empty)'}", flush=True)
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            return False, str(e)
        if wait:
            return self.wait_prompt(timeout)
        time.sleep(0.3)
        return True, None

    def set_mark(self):
        self._mark = len(self.lines)

    def has(self, pattern: str) -> bool:
        text = "\n".join(self.lines)
        if self._buf:
            text += "\n" + self._buf
        return pattern.lower() in text.lower()

    def has_after(self, pattern: str) -> bool:
        lines = self.lines[self._mark:]
        if self._buf:
            lines = lines + [self._buf]
        return any(pattern.lower() in line.lower() for line in lines)

    def drain(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except Empty:
                pass

    def start(self) -> bool:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "main.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=0,
                universal_newlines=True,
                env=env,
                cwd=os.path.dirname(os.path.abspath(__file__)) or "."
            )
        except Exception as e:
            print(f"  [FATAL] {e}")
            return False
        threading.Thread(target=self._reader, daemon=True).start()
        return self.wait_prompt(60)[0]

    def stop(self):
        if self.proc:
            try:
                self.proc.stdin.write("/exit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


TEST_YAML = """defaults:
  model_profile: qwen_cpu
model_profiles:
  qwen_cpu:
    model_name: "qwen2.5:0.5b"
    temperature: 0.0
workflow:
  pattern: supervisor
evolution:
  num_islands: 2
  pop_per_island: 2
  max_generations: 1
research:
  critique_rounds: 1
prompt_templates:
  math_specialist: |
    You are a math specialist. Keep responses short.
  general_assistant: |
    You are a helpful assistant. Keep responses under 50 words.
agents:
  - name: math_agent
    prompt_blocks: ["$math_specialist"]
    tools: [calculator]
    model_profile: qwen_cpu
  - name: general_agent
    prompt_blocks: ["$general_assistant"]
    tools: []
    model_profile: qwen_cpu
"""


def check_prereqs() -> bool:
    print("Checking prerequisites...")
    try:
        r = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            print(f"  Ollama: {r.stdout.strip()}")
        else:
            print("  Ollama: not running")
            return False
    except FileNotFoundError:
        print("  Ollama: not installed")
        return False
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("  Server: not responding")
        return False
        
    r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    if "qwen" in r.stdout.lower():
        print("  Model: found")
    else:
        print("  Pulling qwen2.5:0.5b ...")
        if subprocess.run(["ollama", "pull", "qwen2.5:0.5b"], capture_output=True, text=True, timeout=180).returncode != 0:
            print("  Pull failed")
            return False
            
    for pkg in ["langchain_core", "langchain_openai", "langgraph", "yaml"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"  {pkg}: MISSING")
            return False
            
    with open("_test_agents.yaml", "w") as f:
        f.write(TEST_YAML)
    print("  Prerequisites OK\n")
    return True


def run_tests() -> List[Result]:
    results: List[Result] = []
    t = Tester()
    
    def record(name: str, passed: bool, desc: str, detail: str = ""):
        results.append(Result(name, passed, desc, detail))
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {desc}")
        if detail and not passed:
            print(f"          -> {detail}")

    print("\n[1] Startup")
    ok = t.start()
    record("startup", ok, "starts", "timeout" if not ok else "")
    record("circuit", t.has("Circuit breaker"), "shows CB state")
    
    print("\n[2] Load Config")
    t.drain()
    t.send("/load _test_agents.yaml", timeout=30)
    record("load", t.has("Configuration loaded"), "loads YAML")
    
    print("\n[3] Security Validations")
    t.drain()
    t.send("/load ../etc/passwd", wait=False)
    time.sleep(0.5)
    record("path_block", t.has("Path traversal"), "blocks path traversal")
    
    t.drain()
    t.send("/evolve ", wait=False)
    time.sleep(0.5)
    record("empty_task", t.has("Invalid task"), "blocks empty task")
    
    print("\n[4] Chat Routing & Logic")
    t.drain()
    t.set_mark()
    t.send("What is 2+2?", timeout=90)
    record("math_route", t.has_after("[math_agent]"), "routes to math_agent")
    record("math_ans", t.has_after("4") or t.has_after("[Tool]"), "returns 4")
    
    t.drain()
    t.set_mark()
    t.send("What color is the sky?", timeout=90)
    record("gen_route", t.has_after("[general_agent]"), "routes to general_agent")
    record("sky_blue", t.has_after("blue"), "says blue")
    
    print("\n[5] AlphaEvolve")
    t.drain()
    t.set_mark()
    t.send("/evolve Write a Python function to check if a number is prime", timeout=180)
    record("evo_done", t.has_after("Complete"), "completes evolution")
    record("evo_code", t.has_after("def solve"), "generates code block")
    
    print("\n[6] Clean Exit")
    t.send("/exit", wait=False)
    time.sleep(2)
    record("exit", t.proc.poll() is not None, "exits cleanly")
    t.stop()
    
    return results


def summary(results: List[Result]) -> bool:
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    passed = sum(r.passed for r in results)
    total = len(results)
    
    for r in results:
        symbol = "+" if r.passed else "-"
        print(f"  {symbol} {r.name}: {r.desc}")
        
    print("-" * 60)
    print(f"  TOTAL: {passed}/{total}")
    
    if passed < total:
        print("\n  FAILED:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}: {r.detail or r.desc}")
                
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    if not check_prereqs():
        sys.exit(1)
        
    results = run_tests()
    ok = summary(results)
    
    try:
        os.remove("_test_agents.yaml")
    except Exception:
        pass
        
    sys.exit(0 if ok else 1)