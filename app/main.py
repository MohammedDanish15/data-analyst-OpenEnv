"""
FastAPI server exposing the OpenEnv-compatible REST API:
  POST /reset?task_id=task1
  POST /step?task_id=task1   body: Action JSON
  GET  /state?task_id=task1
  GET  /tasks
  GET  /health
"""
from __future__ import annotations

from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .environment import TASK_CONFIGS, DataAnalystEnv
from .models import Action

app = FastAPI(
    title       = "Data Analyst RL Environment",
    description = "OpenEnv-compatible environment for e-commerce data analysis tasks.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# One env instance per task_id, created on first reset
_envs: Dict[str, DataAnalystEnv] = {}


def _get_env(task_id: str) -> DataAnalystEnv:
    if task_id not in _envs:
        raise HTTPException(
            status_code = 400,
            detail      = f"Environment '{task_id}' not initialised. Call POST /reset?task_id={task_id} first.",
        )
    return _envs[task_id]


@app.post("/reset")
async def reset(task_id: str = Query("task1")) -> Dict[str, Any]:
    """Reset (or create) the environment for the given task and return the initial observation."""
    if task_id not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_CONFIGS)}")
    env = DataAnalystEnv(task_id)
    _envs[task_id] = env
    obs = env.reset()
    return obs.dict()


@app.post("/step")
async def step(action: Action, task_id: str = Query("task1")) -> Dict[str, Any]:
    """Execute one action and return the StepResult."""
    env    = _get_env(task_id)
    result = env.step(action)
    return result.dict()


@app.get("/state")
async def state(task_id: str = Query("task1")) -> Dict[str, Any]:
    """Return the current state without taking an action."""
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all available tasks and their configuration."""
    return {"tasks": list(TASK_CONFIGS.keys()), "configs": TASK_CONFIGS}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
