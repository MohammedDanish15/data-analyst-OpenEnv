"""
inference.py — Baseline agent for the Data Analyst RL Environment.

Mandatory stdout format per OpenEnv spec:
  [START] task=<task_id> env=data-analyst-env model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL  — LLM router base URL  (default: https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier     (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      — HuggingFace API key  (required)
  ENV_URL       — running server URL   (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

TASKS              = ["task1", "task2", "task3"]
MAX_STEPS          = 25
SUCCESS_THRESHOLD  = 0.6   # Fix 5: raised from 0.5 — must earn a real grade
TEMPERATURE        = 0.2
MAX_TOKENS         = 512

# ---------------------------------------------------------------------------
# Stdout loggers (OpenEnv mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=data-analyst-env model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_str  = error if error else "null"
    done_str = str(done).lower()
    # Sanitise action string: no newlines
    action_safe = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_str} error={err_str}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Environment client (HTTP)
# ---------------------------------------------------------------------------

class EnvClient:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = requests.post(f"{self.base}/reset", params={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, action_type: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        payload = {"action_type": action_type, "params": params or {}}
        r = requests.post(
            f"{self.base}/step",
            params  = {"task_id": task_id},
            json    = payload,
            timeout = 30,
        )
        r.raise_for_status()
        return r.json()

    def state(self, task_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base}/state", params={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return r.json()

# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data analyst agent operating in a reinforcement learning environment.
You must choose ONE action per turn from the allowed action types and return it as JSON.

ACTION TYPES (choose one):
  Exploration:   view_head | view_statistics | view_column_types | view_correlations
  Cleaning:      fill_missing_mean | fill_missing_median | drop_missing_rows |
                 detect_outliers   | remove_outliers
  Transformation:normalize | standardize | encode_categorical | apply_minmax_scaling
  Analysis:      find_top_correlations | generate_summary | generate_insights
  Completion:    submit_report | end_episode

For generate_insights, params must be:
  {"insights": [{"finding": "...", "metric": "...", "value": ...}, ...]}

For fill_missing_mean/median or remove_outliers, you may pass:
  {"column": "column_name"}  OR omit params to process all numeric columns.

RESPONSE FORMAT — return ONLY valid JSON, nothing else:
{
  "action_type": "<one of the above>",
  "params": {}
}
""").strip()


def build_user_prompt(step: int, obs: Dict[str, Any], history: List[str]) -> str:
    meta    = obs.get("dataset_metadata", {})
    health  = obs.get("environment_health", {})
    prog    = obs.get("agent_progress", {})
    last    = obs.get("last_action_result", {})

    hist_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step}/{MAX_STEPS}
        Task: {obs.get('task_id')} | Stage: {obs.get('task_stage')}
        Description: {obs.get('task_description')}

        Dataset: {meta.get('total_rows')} rows × {meta.get('total_columns')} cols
        Missing values: {meta.get('missing_counts')}
        Column types: {meta.get('column_types')}
        Data quality score: {health.get('data_quality_score')}
        Invalid actions so far: {health.get('invalid_action_count')}

        Progress: {prog.get('progress_percentage')}% | Intermediate score: {prog.get('intermediate_score')}
        Last action result: {json.dumps(last)[:300]}

        Recent history:
        {hist_block}

        Choose your next action. Return ONLY JSON.
    """).strip()


def get_action(client: OpenAI, step: int, obs: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        resp = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] LLM error at step {step}: {exc}", flush=True)
        # Fallback: safe exploration action
        fallback_actions = [
            "view_head", "view_statistics", "fill_missing_mean",
            "detect_outliers", "remove_outliers", "generate_summary", "submit_report",
        ]
        return {"action_type": fallback_actions[step % len(fallback_actions)], "params": {}}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env: EnvClient, llm: OpenAI, task_id: str) -> Dict[str, Any]:
    log_start(task=task_id, model=MODEL_NAME)

    obs          = env.reset(task_id)
    rewards      : List[float] = []
    history      : List[str]   = []
    steps_taken  = 0
    done         = False
    final_reward = 0.0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        action_dict = get_action(llm, step, obs, history)
        action_type = action_dict.get("action_type", "view_head")
        params      = action_dict.get("params", {})

        try:
            result = env.step(task_id, action_type, params)
        except Exception as exc:
            print(f"[DEBUG] env.step error: {exc}", flush=True)
            log_step(step, action_type, 0.0, False, str(exc))
            rewards.append(0.0)
            steps_taken = step
            continue

        reward = float(result.get("reward", 0.0))
        done   = bool(result.get("done", False))
        info   = result.get("info", {})
        error  = info.get("error", None)
        obs    = result.get("observation", obs)

        rewards.append(reward)
        steps_taken = step
        final_reward = reward  # last step reward (submit_report gives final grade)

        log_step(step, action_type, reward, done, error)

        history.append(
            f"Step {step}: {action_type}(params={json.dumps(params)[:60]}) "
            f"→ reward={reward:.2f}"
        )

        if done:
            break

    # Score = final submit_report reward (the actual grader score 0-1).
    # mean-of-steps dilutes the real grade with small exploration rewards.
    if done and final_reward > 0.0:
        score = float(min(max(final_reward, 0.0), 1.0))
    else:
        score = float(min(max(sum(rewards) / max(len(rewards), 1), 0.0), 1.0))
    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def wait_for_server(base_url: str, retries: int = 30, delay: float = 2.0) -> None:
    for i in range(retries):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"[DEBUG] Server ready at {base_url}", flush=True)
                return
        except Exception:
            pass
        print(f"[DEBUG] Waiting for server… ({i+1}/{retries})", flush=True)
        time.sleep(delay)
    raise RuntimeError(f"Server at {base_url} did not start in time.")


def main() -> None:
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is not set.", flush=True)
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = EnvClient(ENV_URL)

    # If ENV_URL is localhost, try to start the server automatically
    if "localhost" in ENV_URL or "127.0.0.1" in ENV_URL:
        try:
            requests.get(f"{ENV_URL}/health", timeout=3)
        except Exception:
            print("[DEBUG] Starting local server…", flush=True)
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "0.0.0.0", "--port", "7860"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wait_for_server(ENV_URL)

    results = []
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Starting task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            result = run_episode(env, llm, task_id)
            results.append(result)
        except Exception as exc:
            print(f"[ERROR] Task {task_id} failed: {exc}", flush=True)
            results.append({"task_id": task_id, "score": 0.0, "success": False})

    # Final summary
    print("\n" + "="*60, flush=True)
    print("BASELINE RESULTS", flush=True)
    print("="*60, flush=True)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task_id']:8s}  score={r.get('score', 0):.3f}", flush=True)
    overall = sum(r.get("score", 0) for r in results) / len(results)
    print(f"\n  Overall mean score: {overall:.3f}", flush=True)


if __name__ == "__main__":
    main()
