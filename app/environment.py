"""
Core DataAnalystEnv — implements OpenEnv spec:
  reset()  → Observation
  step()   → StepResult
  state()  → dict
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .datasets import (
    generate_task1_dataset,
    generate_task2_dataset,
    generate_task3_dataset,
)
from .graders import grade_task1, grade_task2, grade_task3
from .models import (
    Action,
    ActionType,
    AgentProgress,
    DatasetMetadata,
    EnvironmentHealth,
    Observation,
    StepResult,
    TaskStage,
)

# ---------------------------------------------------------------------------
# Task configuration registry
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task1": {
        "name":        "Missing Value Fixing",
        "description": (
            "Explore the e-commerce dataset. Identify and fill missing values in "
            "the 'age' and 'total_price' columns, then submit your report."
        ),
        "difficulty": "easy",
        "max_steps":  25,
        "seed":       42,
    },
    "task2": {
        "name":        "Full Cleaning + Transformation",
        "description": (
            "Clean all missing values, remove impossible outliers (age > 100, negative "
            "quantities), fix inconsistent gender labels, encode all categorical columns, "
            "apply normalization, and submit your report."
        ),
        "difficulty": "medium",
        "max_steps":  25,
        "seed":       123,
    },
    "task3": {
        "name":        "Full Analysis + Insight Generation",
        "description": (
            "Perform the complete cleaning pipeline, discover hidden correlations "
            "(discount vs return_flag, Electronics vs return rate, CLV vs spend), "
            "and submit a structured report with ≥3 JSON insights: "
            '{"finding": str, "metric": str, "value": float|str}.'
        ),
        "difficulty": "hard",
        "max_steps":  25,
        "seed":       456,
    },
}

# Reward table — keeps step rewards in [0, 1] per action category
STEP_REWARDS = {
    "exploration_new":    0.05,   # first time exploring a view
    "exploration_repeat": 0.02,   # repeated exploration (diminishing)
    "cleaning_effective": 0.15,   # filling / dropping actually changed data
    "cleaning_noop":      0.00,   # cleaning action but nothing changed
    "outlier_detect":     0.05,
    "outlier_remove":     0.15,
    "transform":          0.10,
    "analysis":           0.10,
    "insight_record":     0.20,
    "invalid":            0.00,
}

# Max invalid actions before forced termination
MAX_INVALID = 5


class DataAnalystEnv:
    """OpenEnv-compatible Data Analyst environment."""

    def __init__(self, task_id: str = "task1") -> None:
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"task_id must be one of {list(TASK_CONFIGS)}")
        self.task_id = task_id
        self.cfg     = TASK_CONFIGS[task_id]
        self.df: Optional[pd.DataFrame] = None
        self._init_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        self.step_number          = 0
        self.invalid_action_count = 0
        self.done                 = False
        self.current_stage        = TaskStage.EXPLORATION
        self.action_log:          List[str]            = []
        self.submitted_insights:  List[Dict[str, Any]] = []
        self.last_action_result:  Dict[str, Any]       = {}

        # Progress flags (used for intermediate_score)
        self.f_explored        = False
        self.f_filled_missing  = False
        self.f_outliers_removed = False
        self.f_encoded         = False
        self.f_transformed     = False
        self.f_analyzed        = False
        self.f_insights        = False

        # Track actions already taken (for diminishing exploration reward)
        self._seen_exploration: set = set()

        # Fix 4 — repeated action penalty: track previous action type
        self._last_action_type: Optional[str] = None

    def _load_dataset(self) -> pd.DataFrame:
        seed = self.cfg["seed"]
        if self.task_id == "task1":
            return generate_task1_dataset(seed)
        elif self.task_id == "task2":
            return generate_task2_dataset(seed)
        else:
            return generate_task3_dataset(seed)

    def _quality_score(self) -> float:
        """Composite 0-1 data-quality metric."""
        if self.df is None or self.df.empty:
            return 0.0
        total      = self.df.shape[0] * self.df.shape[1]
        missing    = int(self.df.isnull().sum().sum())
        miss_score = 1.0 - missing / max(total, 1)

        outlier_score = 1.0
        if "age" in self.df.columns:
            age_num  = pd.to_numeric(self.df["age"], errors="coerce")
            bad_age  = int(((age_num > 100) | (age_num < 0)).sum())
            outlier_score = max(0.0, 1.0 - bad_age / max(len(self.df), 1))

        return round((miss_score + outlier_score) / 2, 4)

    def _intermediate_score(self) -> float:
        score = 0.0
        if self.f_explored:        score += 0.10
        if self.f_filled_missing:  score += 0.25
        if self.f_outliers_removed:score += 0.20
        if self.f_encoded:         score += 0.15
        if self.f_transformed:     score += 0.15
        if self.f_analyzed:        score += 0.10
        if self.f_insights:        score += 0.05
        return min(round(score, 4), 1.0)

    def _build_metadata(self) -> DatasetMetadata:
        missing_counts  = {c: int(self.df[c].isnull().sum()) for c in self.df.columns}
        column_types    = {c: str(self.df[c].dtype)          for c in self.df.columns}
        numeric_summary: Dict[str, Dict[str, float]] = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            d = self.df[col].describe()
            numeric_summary[col] = {
                "mean": round(float(d.get("mean", 0) or 0), 4),
                "std":  round(float(d.get("std",  0) or 0), 4),
                "min":  round(float(d.get("min",  0) or 0), 4),
                "max":  round(float(d.get("max",  0) or 0), 4),
            }
        return DatasetMetadata(
            total_rows      = len(self.df),
            total_columns   = len(self.df.columns),
            missing_counts  = missing_counts,
            numeric_summary = numeric_summary,
            column_types    = column_types,
        )

    def _build_obs(self) -> Observation:
        progress = min(self.step_number / self.cfg["max_steps"], 1.0)
        return Observation(
            dataset_metadata  = self._build_metadata(),
            task_stage        = self.current_stage,
            environment_health= EnvironmentHealth(
                step_number          = self.step_number,
                invalid_action_count = self.invalid_action_count,
                data_quality_score   = self._quality_score(),
            ),
            agent_progress    = AgentProgress(
                progress_percentage = round(progress * 100, 2),
                intermediate_score  = self._intermediate_score(),
            ),
            last_action_result = self.last_action_result,
            task_id            = self.task_id,
            task_description   = self.cfg["description"],
        )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment; returns initial Observation."""
        self._init_state()
        self.df = self._load_dataset()
        return self._build_obs()

    def step(self, action: Action) -> StepResult:
        """Execute action; returns StepResult."""
        if self.done:
            return StepResult(
                observation = self._build_obs(),
                reward      = 0.0,
                done        = True,
                info        = {"error": "Episode already finished. Call reset()."},
            )

        self.step_number += 1
        error_msg: Optional[str] = None

        try:
            reward, result_info = self._dispatch(action)

            # Fix 4 — repeated action penalty: halve reward if same action as last step.
            # Exemptions: submit_report and end_episode are never penalised.
            _exempt = {ActionType.SUBMIT_REPORT, ActionType.END_EPISODE}
            if (
                action.action_type not in _exempt
                and self._last_action_type == action.action_type.value
            ):
                reward *= 0.5
                result_info["repeat_penalty"] = True

            self._last_action_type  = action.action_type.value
            self.last_action_result = result_info
            self.action_log.append(action.action_type.value)
        except Exception as exc:
            reward             = 0.0
            error_msg          = str(exc)
            self.last_action_result = {"error": error_msg}
            self.invalid_action_count += 1

        # Termination checks
        if self.step_number >= self.cfg["max_steps"]:
            self.done = True
        if self.invalid_action_count >= MAX_INVALID:
            self.done = True

        info: Dict[str, Any] = {
            "step":       self.step_number,
            "stage":      self.current_stage.value,
            "action_log": self.action_log,
        }
        if error_msg:
            info["error"] = error_msg

        return StepResult(
            observation = self._build_obs(),
            reward      = float(np.clip(reward, 0.0, 1.0)),
            done        = self.done,
            info        = info,
        )

    def state(self) -> Dict[str, Any]:
        """Return current state as plain dict (OpenEnv spec)."""
        return self._build_obs().dict()

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        at     = action.action_type
        params = action.params or {}

        # ── EXPLORATION ──────────────────────────────────────────────
        if at == ActionType.VIEW_HEAD:
            new   = at.value not in self._seen_exploration
            r     = STEP_REWARDS["exploration_new"] if new else STEP_REWARDS["exploration_repeat"]
            self._seen_exploration.add(at.value)
            self.f_explored = True
            self.current_stage = TaskStage.EXPLORATION
            head = self.df.head(5).where(pd.notnull(self.df.head(5)), None).to_dict(orient="records")
            return r, {"head": head}

        elif at == ActionType.VIEW_STATISTICS:
            new = at.value not in self._seen_exploration
            r   = STEP_REWARDS["exploration_new"] if new else STEP_REWARDS["exploration_repeat"]
            self._seen_exploration.add(at.value)
            self.f_explored = True
            self.current_stage = TaskStage.EXPLORATION
            stats = (self.df.describe().round(4)
                         .where(pd.notnull(self.df.describe()), None)
                         .to_dict())
            return r, {"statistics": stats}

        elif at == ActionType.VIEW_COLUMN_TYPES:
            new = at.value not in self._seen_exploration
            r   = STEP_REWARDS["exploration_new"] if new else STEP_REWARDS["exploration_repeat"]
            self._seen_exploration.add(at.value)
            self.f_explored = True
            self.current_stage = TaskStage.EXPLORATION
            return r, {
                "column_types":  {c: str(self.df[c].dtype) for c in self.df.columns},
                "missing_counts":{c: int(self.df[c].isnull().sum()) for c in self.df.columns},
                "sample_values": {c: str(self.df[c].dropna().iloc[0])
                                  if not self.df[c].dropna().empty else "all_null"
                                  for c in self.df.columns},
            }

        elif at == ActionType.VIEW_CORRELATIONS:
            num_df = self.df.select_dtypes(include=[np.number])
            if num_df.shape[1] < 2:
                self.invalid_action_count += 1
                return 0.0, {"error": "Need ≥2 numeric columns for correlation."}
            new = at.value not in self._seen_exploration
            r   = STEP_REWARDS["exploration_new"] if new else STEP_REWARDS["exploration_repeat"]
            self._seen_exploration.add(at.value)
            self.f_explored = True
            self.current_stage = TaskStage.EXPLORATION
            corr = num_df.corr().round(4).where(pd.notnull(num_df.corr()), None).to_dict()
            return r, {"correlations": corr}

        # ── CLEANING ─────────────────────────────────────────────────
        elif at == ActionType.FILL_MISSING_MEAN:
            col    = params.get("column")
            filled = self._fill_missing(col, method="mean")
            if filled > 0:
                self.f_filled_missing = True
                self.current_stage    = TaskStage.CLEANING
                return STEP_REWARDS["cleaning_effective"], {"cells_filled": filled, "method": "mean"}
            return STEP_REWARDS["cleaning_noop"], {"cells_filled": 0, "method": "mean"}

        elif at == ActionType.FILL_MISSING_MEDIAN:
            col    = params.get("column")
            filled = self._fill_missing(col, method="median")
            if filled > 0:
                self.f_filled_missing = True
                self.current_stage    = TaskStage.CLEANING
                return STEP_REWARDS["cleaning_effective"], {"cells_filled": filled, "method": "median"}
            return STEP_REWARDS["cleaning_noop"], {"cells_filled": 0, "method": "median"}

        elif at == ActionType.DROP_MISSING_ROWS:
            before = len(self.df)
            self.df.dropna(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            dropped = before - len(self.df)
            if dropped > 0:
                self.f_filled_missing = True
                self.current_stage    = TaskStage.CLEANING
                return 0.10, {"rows_dropped": dropped}   # slightly lower — loses data
            return STEP_REWARDS["cleaning_noop"], {"rows_dropped": 0}

        elif at == ActionType.DETECT_OUTLIERS:
            result: Dict[str, int] = {}
            for c in self.df.select_dtypes(include=[np.number]).columns:
                Q1, Q3  = self.df[c].quantile(0.25), self.df[c].quantile(0.75)
                IQR     = Q3 - Q1
                bad     = int(((self.df[c] < Q1 - 1.5*IQR) | (self.df[c] > Q3 + 1.5*IQR)).sum())
                if bad > 0:
                    result[c] = bad
            self.current_stage = TaskStage.CLEANING
            r = STEP_REWARDS["outlier_detect"] if result else STEP_REWARDS["exploration_repeat"]
            return r, {"outliers_detected": result}

        elif at == ActionType.REMOVE_OUTLIERS:
            col     = params.get("column")
            removed = self._remove_outliers(col)
            if removed > 0:
                self.f_outliers_removed = True
                self.current_stage      = TaskStage.CLEANING
                return STEP_REWARDS["outlier_remove"], {"rows_removed": removed}
            self.invalid_action_count += 1
            return STEP_REWARDS["cleaning_noop"], {"rows_removed": 0, "message": "No outliers remain to remove"}

        # ── TRANSFORMATION ───────────────────────────────────────────
        elif at == ActionType.NORMALIZE:
            cols = self._normalize_cols()
            self.f_transformed = True
            self.current_stage = TaskStage.TRANSFORMATION
            return STEP_REWARDS["transform"], {"normalized_columns": cols}

        elif at == ActionType.STANDARDIZE:
            cols = self._standardize_cols()
            self.f_transformed = True
            self.current_stage = TaskStage.TRANSFORMATION
            return STEP_REWARDS["transform"], {"standardized_columns": cols}

        elif at == ActionType.ENCODE_CATEGORICAL:
            cols = self._encode_categoricals()
            if cols:
                self.f_encoded     = True
                self.current_stage = TaskStage.TRANSFORMATION
                return STEP_REWARDS["transform"], {"encoded_columns": cols}
            return STEP_REWARDS["cleaning_noop"], {"message": "No object columns to encode."}

        elif at == ActionType.APPLY_MINMAX_SCALING:
            cols = self._minmax_cols()
            self.f_transformed = True
            self.current_stage = TaskStage.TRANSFORMATION
            return STEP_REWARDS["transform"], {"scaled_columns": cols}

        # ── ANALYSIS ─────────────────────────────────────────────────
        elif at == ActionType.FIND_TOP_CORRELATIONS:
            num_df = self.df.select_dtypes(include=[np.number])
            if num_df.shape[1] < 2:
                self.invalid_action_count += 1
                return 0.0, {"error": "Need ≥2 numeric columns."}
            top = self._top_correlations(num_df)
            self.f_analyzed    = True
            self.current_stage = TaskStage.ANALYSIS
            return STEP_REWARDS["analysis"], {"top_correlations": top}

        elif at == ActionType.GENERATE_SUMMARY:
            summary = {
                "total_rows":           len(self.df),
                "total_columns":        len(self.df.columns),
                "remaining_missing":    int(self.df.isnull().sum().sum()),
                "numeric_columns":      len(self.df.select_dtypes(include=[np.number]).columns),
                "categorical_columns":  len(self.df.select_dtypes(include=["object"]).columns),
                "data_quality_score":   self._quality_score(),
                "actions_taken":        len(self.action_log),
            }
            self.f_analyzed    = True
            self.current_stage = TaskStage.ANALYSIS
            return STEP_REWARDS["analysis"], {"summary": summary}

        elif at == ActionType.GENERATE_INSIGHTS:
            insights: List[Dict[str, Any]] = params.get("insights", [])
            if not insights:
                self.invalid_action_count += 1
                return 0.0, {
                    "error": (
                        'Provide insights in params: {"insights": ['
                        '{"finding": str, "metric": str, "value": float|str}, ...]}'
                    )
                }
            self.submitted_insights = insights
            self.f_insights    = True
            self.f_analyzed    = True
            self.current_stage = TaskStage.ANALYSIS
            return STEP_REWARDS["insight_record"], {"insights_recorded": len(insights)}

        # ── COMPLETION ───────────────────────────────────────────────
        elif at == ActionType.SUBMIT_REPORT:
            # Allow inline insights via params
            inline = params.get("insights", [])
            if inline:
                self.submitted_insights = inline

            if self.task_id == "task1":
                grade = grade_task1(self.df, self.action_log)
            elif self.task_id == "task2":
                grade = grade_task2(self.df, self.action_log)
            else:
                grade = grade_task3(self.df, self.action_log, self.submitted_insights)

            self.done          = True
            self.current_stage = TaskStage.REPORTING
            return float(grade["final_reward"]), {"grade": grade, "message": "Episode complete."}

        elif at == ActionType.END_EPISODE:
            self.done = True
            return 0.0, {"message": "Agent ended episode early."}

        else:
            self.invalid_action_count += 1
            return 0.0, {"error": f"Unrecognised action: {at}"}

    # ------------------------------------------------------------------
    # Private mutators
    # ------------------------------------------------------------------

    def _fill_missing(self, col: Optional[str], method: str) -> int:
        def _fill_series(s: pd.Series) -> int:
            before = int(s.isnull().sum())
            if before == 0:
                return 0
            fill_val = s.mean() if method == "mean" else s.median()
            self.df[s.name] = s.fillna(fill_val)
            return before - int(self.df[s.name].isnull().sum())

        if col:
            if col not in self.df.columns:
                self.invalid_action_count += 1
                raise ValueError(f"Column '{col}' not found.")
            if self.df[col].dtype not in [np.float64, np.int64, float, int]:
                raise ValueError(f"Column '{col}' is not numeric.")
            return _fill_series(self.df[col])
        else:
            return sum(_fill_series(self.df[c])
                       for c in self.df.select_dtypes(include=[np.number]).columns)

    def _remove_outliers(self, col: Optional[str]) -> int:
        before = len(self.df)

        def _remove_col(c: str) -> None:
            Q1, Q3 = self.df[c].quantile(0.25), self.df[c].quantile(0.75)
            IQR    = Q3 - Q1
            mask   = (self.df[c] >= Q1 - 1.5*IQR) & (self.df[c] <= Q3 + 1.5*IQR)
            self.df = self.df[mask].copy()   # .copy() prevents read-only numpy views

        if col:
            if col not in self.df.columns:
                self.invalid_action_count += 1
                raise ValueError(f"Column '{col}' not found.")
            _remove_col(col)
        else:
            for c in self.df.select_dtypes(include=[np.number]).columns:
                _remove_col(c)

        self.df.reset_index(drop=True, inplace=True)
        return before - len(self.df)

    def _normalize_cols(self) -> List[str]:
        cols = list(self.df.select_dtypes(include=[np.number]).columns)
        for c in cols:
            lo, hi = self.df[c].min(), self.df[c].max()
            if hi - lo > 0:
                self.df[c] = (self.df[c] - lo) / (hi - lo)
        return cols

    def _standardize_cols(self) -> List[str]:
        cols = list(self.df.select_dtypes(include=[np.number]).columns)
        for c in cols:
            mu, sigma = self.df[c].mean(), self.df[c].std()
            if sigma > 0:
                self.df[c] = (self.df[c] - mu) / sigma
        return cols

    def _encode_categoricals(self) -> List[str]:
        cats = [c for c in self.df.select_dtypes(include=["object"]).columns
                if c != "customer_id"]
        for c in cats:
            self.df[c] = pd.Categorical(self.df[c]).codes.astype(float)
        return cats

    def _minmax_cols(self) -> List[str]:
        cols = list(self.df.select_dtypes(include=[np.number]).columns)
        for c in cols:
            lo, hi = self.df[c].min(), self.df[c].max()
            if hi - lo > 0:
                self.df[c] = (self.df[c] - lo) / (hi - lo)
        return cols

    @staticmethod
    def _top_correlations(num_df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
        # Use .copy() so fill_diagonal doesn't hit a read-only numpy view
        # (happens when df has been sliced/normalized before this call)
        corr_matrix = num_df.corr().abs()
        corr = pd.DataFrame(
            corr_matrix.values.copy(),
            index   = corr_matrix.index,
            columns = corr_matrix.columns,
        )
        np.fill_diagonal(corr.values, 0)
        pairs = []
        tmp   = corr.copy()
        for _ in range(n):
            if tmp.max().max() == 0:
                break
            idx = tmp.stack().idxmax()
            pairs.append({
                "col1":        idx[0],
                "col2":        idx[1],
                "correlation": round(float(tmp.loc[idx[0], idx[1]]), 4),
            })
            tmp.loc[idx[0], idx[1]] = 0
            tmp.loc[idx[1], idx[0]] = 0
        return pairs
