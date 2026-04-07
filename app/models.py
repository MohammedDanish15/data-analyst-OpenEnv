"""
Typed Pydantic models for the Data Analyst RL Environment.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskStage(str, Enum):
    EXPLORATION    = "exploration"
    CLEANING       = "cleaning"
    TRANSFORMATION = "transformation"
    ANALYSIS       = "analysis"
    REPORTING      = "reporting"


class ActionType(str, Enum):
    VIEW_HEAD             = "view_head"
    VIEW_STATISTICS       = "view_statistics"
    VIEW_COLUMN_TYPES     = "view_column_types"
    VIEW_CORRELATIONS     = "view_correlations"
    FILL_MISSING_MEAN     = "fill_missing_mean"
    FILL_MISSING_MEDIAN   = "fill_missing_median"
    DROP_MISSING_ROWS     = "drop_missing_rows"
    DETECT_OUTLIERS       = "detect_outliers"
    REMOVE_OUTLIERS       = "remove_outliers"
    NORMALIZE             = "normalize"
    STANDARDIZE           = "standardize"
    ENCODE_CATEGORICAL    = "encode_categorical"
    APPLY_MINMAX_SCALING  = "apply_minmax_scaling"
    FIND_TOP_CORRELATIONS = "find_top_correlations"
    GENERATE_SUMMARY      = "generate_summary"
    GENERATE_INSIGHTS     = "generate_insights"
    SUBMIT_REPORT         = "submit_report"
    END_EPISODE           = "end_episode"


class DatasetMetadata(BaseModel):
    total_rows:      int
    total_columns:   int
    missing_counts:  Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]
    column_types:    Dict[str, str]


class EnvironmentHealth(BaseModel):
    step_number:          int
    invalid_action_count: int
    data_quality_score:   float = Field(ge=0.0, le=1.0)


class AgentProgress(BaseModel):
    progress_percentage: float = Field(ge=0.0, le=100.0)
    intermediate_score:  float = Field(ge=0.0, le=1.0)


class Observation(BaseModel):
    dataset_metadata:   DatasetMetadata
    task_stage:         TaskStage
    environment_health: EnvironmentHealth
    agent_progress:     AgentProgress
    last_action_result: Optional[Dict[str, Any]] = None
    task_id:            str
    task_description:   str


class Action(BaseModel):
    action_type: ActionType
    params:      Optional[Dict[str, Any]] = None


class Reward(BaseModel):
    value:  float = Field(ge=0.0, le=1.0)
    reason: str


class StepResult(BaseModel):
    observation: Observation
    reward:      float = Field(ge=0.0, le=1.0)
    done:        bool
    info:        Dict[str, Any] = {}
