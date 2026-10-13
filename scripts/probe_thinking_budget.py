"""Probe: how many tokens does thinking eat on hard MATH-style problems?

Decides `max_tokens_per_node` in configs/e0.yaml. Writes runs/probe/thinking_budget.json.
"""
import os, json, urllib.request, concurrent.futures as cf, pathlib

BASE = "http://aimaster-llm-gateway-service.aimaster-web:8080/llm/v1"
KEY = os.environ["LLM_GATEWAY_API_KEY"]

PROBS = [
    r"Let $f(x)=x^3-3x+1$. Find the sum of all real $r$ with $f(f(r))=r$ but $f(r)\neq r$. Put the answer in \boxed{}.",
    r"How many ordered triples $(a,b,c)$ of positive integers satisfy $\mathrm{lcm}(a,b)=72$, $\mathrm{lcm}(a,c)=24$, $\mathrm{lcm}(b,c)=18$? Put the answer in \boxed{}.",
    r"Find the number of positive integers $n\le 1000$ such that $\lfloor\sqrt{n}\rfloor$ divides $n$. Put the answer in \boxed{}.",
]
MODELS = ["DeepSeek", "GLM-5.3-Flash", "gpt-oss-120b", "gemma-4-31B-it", "Laguna-S-2.1"]
CAPS = [1024, 8192]


def one(model, max_tokens, prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        BASE + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=900))
        ch, us = d["choices"][0], d["usage"]
        txt = ch["message"].get("content") or ""
        return dict(model=model, cap=max_tokens, finish=ch.get("finish_reason"),
                    prompt_tokens=us["prompt_tokens"], completion_tokens=us["completion_tokens"],
                    reasoning_tokens=(us.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    has_boxed="\\boxed" in txt, empty=not txt.strip(), chars=len(txt))
    except Exception as exc:
        return dict(model=model, cap=max_tokens, error=f"{type(exc).__name__}: {exc}"[:200])


jobs = [(m, c, p) for m in MODELS for c in CAPS for p in PROBS]
with cf.ThreadPoolExecutor(15) as ex:
    rows = list(ex.map(lambda a: one(*a), jobs))

out = pathlib.Path("runs/probe/thinking_budget.json")
out.write_text(json.dumps(rows, indent=1))
for m in MODELS:
    for c in CAPS:
        rs = [r for r in rows if r["model"] == m and r["cap"] == c]
        ok = [r for r in rs if "error" not in r]
        print(f"{m:16s} cap={c:5d} ok={len(ok)}/{len(rs)} "
              f"truncated={sum(r['finish']=='length' for r in ok)} "
              f"empty={sum(r['empty'] for r in ok)} boxed={sum(r['has_boxed'] for r in ok)} "
              f"comp={[r['completion_tokens'] for r in ok]} reas={[r['reasoning_tokens'] for r in ok]}")
        for r in rs:
            if "error" in r:
                print("    ", r["error"])
print("wrote", out)
