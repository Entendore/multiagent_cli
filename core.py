# tools.py
"""Built-in tools with security improvements."""
import re
from langchain_core.tools import tool
from core import SafeCalculator, CodeSandbox

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression safely (no eval)."""
    valid, result = SafeCalculator.evaluate(expression)
    return result

@tool
def python_repl(code: str) -> str:
    """Execute Python code in a sandboxed environment."""
    sandbox = CodeSandbox(timeout=5.0)
    res = sandbox.execute(code)
    return res.output or "(no output)" if res.success else f"Error: {res.error}"

@tool
def echo(text: str) -> str:
    """Echo the input text back."""
    return text

@tool
def score_solution(solution: str, criteria: str = "clarity, correctness") -> str:
    """Score a solution from 0-10 based on criteria."""
    score = 0.0
    if len(solution) > 50: score += 2
    if len(solution) > 150: score += 2
    if any(kw in solution.lower() for kw in ["step","because","therefore"]): score += 3
    if any(c.strip().lower() in solution.lower() for c in criteria.split(",")): score += 3
    return f"{min(score, 10):.1f}/10"

@tool
def extract_hypotheses(text: str) -> str:
    """Extract hypotheses from text in H1, H2 format."""
    matches = re.findall(r'H\d+\s*[:.]?\s*(.+)', text, re.IGNORECASE)
    if not matches: matches = re.findall(r'\d+\.\s*(.+)', text)
    return "\n".join(f"H{i+1}: {m.strip()}" for i, m in enumerate(matches[:5])) if matches else "None found"

@tool
def format_report(sections: str) -> str:
    """Format text into a report with proper markdown headers."""
    out = []
    for line in sections.strip().split("\n"):
        line = line.strip()
        if not line: continue
        if any(line.upper().startswith(kw) for kw in ["ABSTRACT","INTRO","METHOD","RESULT","DISCUSS","CONCLUSION","HYPOTHESIS"]):
            out.append(f"\n## {line}\n")
        else: out.append(line)
    return "\n".join(out)

def get_builtin_tools() -> list:
    return [calculator, python_repl, echo, score_solution, extract_hypotheses, format_report]