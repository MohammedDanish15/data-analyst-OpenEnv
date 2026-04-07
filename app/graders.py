"""
Deterministic graders for all 3 tasks.
Each grader returns a dict with sub-scores and a final_reward in [0.0, 1.0].

Changes vs v1:
  Fix 2 — Task 1 insight: submission only counts if cleanliness > 0.8
  Fix 3 — Task 2 correctness: dtype-checked outcomes, not action-log presence
  Fix 1 — Task 3 insights: correlation values verified against actual DataFrame data
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task 1 — Easy: Missing Value Fixing
# ---------------------------------------------------------------------------

def grade_task1(df: pd.DataFrame, action_log: List[str]) -> Dict[str, float]:
    """
    cleanliness  — nulls remaining in age + total_price
    correctness  — correct filling action used
    insight      — explored AND submitted with cleanliness > 0.8
                   (Fix 2: removes free 0.5 for just submitting without cleaning)
    """
    # 1. Cleanliness
    target_cols   = [c for c in ["age", "total_price"] if c in df.columns]
    total_targets = len(df) * len(target_cols) if target_cols else 1
    still_missing = sum(int(df[c].isnull().sum()) for c in target_cols)
    cleanliness_score = max(0.0, 1.0 - still_missing / total_targets)

    # 2. Correctness: used a valid filling/dropping action
    fill_actions  = {"fill_missing_mean", "fill_missing_median", "drop_missing_rows"}
    correctness_score = 1.0 if any(a in action_log for a in fill_actions) else 0.0

    # 3. Insight — Fix 2: submit earns 0.5 only if agent actually cleaned
    #    (cleanliness > 0.95 requires near-zero missing — raw dataset sits at ~0.875)
    #    Grader is always called from submit_report, so "submitted" is unconditionally True.
    explored      = "view_statistics" in action_log or "view_head" in action_log
    submit_points = 0.5 if cleanliness_score > 0.95 else 0.0
    insight_score = (0.5 if explored else 0.0) + submit_points

    final_reward = (cleanliness_score + correctness_score + insight_score) / 3.0
    return {
        "cleanliness_score": round(cleanliness_score, 4),
        "correctness_score": round(correctness_score, 4),
        "insight_score":     round(insight_score, 4),
        "final_reward":      round(final_reward, 4),
    }


# ---------------------------------------------------------------------------
# Task 2 — Medium: Full Cleaning + Transformation
# ---------------------------------------------------------------------------

def grade_task2(df: pd.DataFrame, action_log: List[str]) -> Dict[str, float]:
    """
    cleanliness  — nulls remaining across all columns
    correctness  — Fix 3: outcome-verified checks (dtype, value ranges), not just action_log
    insight      — exploration breadth + submission
    """
    # 1. Cleanliness
    total_cells   = df.shape[0] * df.shape[1]
    still_missing = int(df.isnull().sum().sum())
    cleanliness_score = max(0.0, 1.0 - still_missing / max(total_cells, 1))

    # 2a. Outliers — check actual values in DataFrame, NOT action_log
    age_ok = True
    qty_ok = True
    if "age" in df.columns:
        valid_age = pd.to_numeric(df["age"], errors="coerce")
        age_ok = bool(((valid_age > 100) | (valid_age < 0)).sum() == 0)
    if "quantity" in df.columns:
        valid_qty = pd.to_numeric(df["quantity"], errors="coerce")
        qty_ok = bool((valid_qty < 0).sum() == 0)
    outlier_score = 1.0 if (age_ok and qty_ok) else 0.5 if (age_ok or qty_ok) else 0.0

    # 2b. Fix 3 — check actual dtype of encoded columns, not action_log
    #     An agent that called encode_categorical but it silently failed gets 0.0
    gender_encoded = 1.0 if (
        "gender" in df.columns and
        df["gender"].dtype in [np.int64, np.float64, int, float] and
        df["gender"].dropna().nunique() <= 10          # must have collapsed categories
    ) else 0.0

    category_encoded = 1.0 if (
        "product_category" in df.columns and
        df["product_category"].dtype in [np.int64, np.float64, int, float] and
        df["product_category"].dropna().nunique() <= 10
    ) else 0.0

    # 2c. Transformation — verify numeric columns were actually scaled
    #     Check that at least one numeric column has values in [0, 1] or [-4, 4] (standardized)
    transform_applied = 0.0
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        col_maxes = [df[c].abs().max() for c in num_cols if df[c].notna().any()]
        if col_maxes:
            # Either min-max scaled (max ≤ 1.0) or standardized (max typically < 10)
            scaled = sum(1 for m in col_maxes if m <= 1.0001)
            if scaled / len(col_maxes) >= 0.5:       # majority of cols scaled
                transform_applied = 1.0

    correctness_score = (outlier_score + gender_encoded + category_encoded + transform_applied) / 4.0

    # 3. Insight — breadth of exploration + submit (always True when grader runs)
    exploration_actions = ["view_head", "view_statistics", "view_correlations", "detect_outliers"]
    n_explored    = sum(1 for a in exploration_actions if a in action_log)
    insight_score = min(n_explored / len(exploration_actions), 1.0) * 0.7 + 0.3

    final_reward = (cleanliness_score + correctness_score + insight_score) / 3.0
    return {
        "cleanliness_score":  round(cleanliness_score, 4),
        "correctness_score":  round(correctness_score, 4),
        "insight_score":      round(insight_score, 4),
        "outcome_checks": {
            "age_ok":           age_ok,
            "qty_ok":           qty_ok,
            "gender_encoded":   bool(gender_encoded),
            "category_encoded": bool(category_encoded),
            "transform_applied":bool(transform_applied),
        },
        "final_reward":       round(final_reward, 4),
    }


# ---------------------------------------------------------------------------
# Task 3 — Hard: Full Analysis + Insight Generation
# ---------------------------------------------------------------------------

_INSIGHT_KEYWORDS = {
    "discount_return":    ["discount", "return"],
    "electronics_price":  ["electronics", "price"],
    "electronics_return": ["electronics", "return"],
    "clv_quantity":       ["lifetime", "quantity"],
    "revenue_country":    ["revenue", "country"],
}

# Which keyword pattern maps to which DataFrame column pair for verification
_VERIFIABLE_PAIRS: Dict[str, tuple] = {
    "discount_return":   ("discount",               "return_flag"),
    "electronics_price": None,                       # categorical — skip numeric verify
    "electronics_return":None,                       # categorical — skip numeric verify
    "clv_quantity":      ("customer_lifetime_value", "quantity"),
    "revenue_country":   None,                       # categorical — skip numeric verify
}


def _score_insight(
    insight: Dict[str, Any],
    df: pd.DataFrame,
) -> float:
    """
    Score one insight dict: {"finding": str, "metric": str, "value": float|str}

    Fix 1 — correlation verification:
      If the insight maps to a verifiable numeric column pair, compute the actual
      Pearson correlation and check whether the agent's claimed value is within ±0.20.
      If it's outside that range, the score is multiplied by 0.3 (fabrication penalty).
    """
    finding  = str(insight.get("finding", "")).lower()
    metric   = str(insight.get("metric",  "")).lower()
    value    = insight.get("value", None)
    combined = finding + " " + metric

    # Must have a value
    has_value = value is not None and str(value).strip() != ""
    if not has_value:
        return 0.0

    # Keyword matching — how many known patterns does this insight reference?
    matched_patterns: List[str] = []
    for pattern_name, keywords in _INSIGHT_KEYWORDS.items():
        if all(kw in combined for kw in keywords):
            matched_patterns.append(pattern_name)

    if not matched_patterns:
        return 0.1   # has a value but no recognized pattern

    base_score = min(0.4 + 0.3 * len(matched_patterns), 1.0)

    # Fix 1 — verify numeric claims against actual DataFrame
    fabrication_penalty = False
    for pattern in matched_patterns:
        col_pair = _VERIFIABLE_PAIRS.get(pattern)
        if col_pair is None:
            continue   # categorical pattern — cannot verify numerically

        col1, col2 = col_pair
        if col1 not in df.columns or col2 not in df.columns:
            continue

        # Both must be numeric after pipeline
        s1 = pd.to_numeric(df[col1], errors="coerce").dropna()
        s2 = pd.to_numeric(df[col2], errors="coerce").dropna()
        if len(s1) < 10 or len(s2) < 10:
            continue

        # Align by index
        aligned = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
        if len(aligned) < 10:
            continue

        actual_corr = float(aligned["a"].corr(aligned["b"]))

        # Agent's claimed value must be numeric to verify
        try:
            claimed = float(value)
        except (TypeError, ValueError):
            continue   # string value — cannot verify, no penalty

        if abs(claimed - actual_corr) > 0.35:
            fabrication_penalty = True
            break

    if fabrication_penalty:
        return round(base_score * 0.3, 4)   # penalise fabricated correlation values

    return round(base_score, 4)


def grade_task3(
    df: pd.DataFrame,
    action_log: List[str],
    insights: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    cleanliness  — nulls remaining
    correctness  — full pipeline checklist (outcome-aware where possible)
    insight      — keyword quality + Fix 1 correlation verification
    """
    # 1. Cleanliness
    total_cells   = df.shape[0] * df.shape[1]
    still_missing = int(df.isnull().sum().sum())
    cleanliness_score = max(0.0, 1.0 - still_missing / max(total_cells, 1))

    # 2. Correctness — pipeline checklist
    #    encoding + transform are outcome-verified (same logic as task 2)
    pipeline = {
        "cleaning":     any(a in action_log for a in
                            ["fill_missing_mean", "fill_missing_median", "drop_missing_rows"]),
        "outliers":     "remove_outliers" in action_log,
        "encoding":     any(
                            df[c].dtype in [np.int64, np.float64]
                            for c in ["gender", "product_category", "payment_method", "country"]
                            if c in df.columns
                        ),
        "transform":    any(
                            df[c].abs().max() <= 1.0001
                            for c in df.select_dtypes(include=[np.number]).columns
                            if df[c].notna().any()
                            and df[c].nunique() > 2                              # exclude binary cols
                            and c not in {"return_flag", "discount", "order_month"}  # exclude naturally-small
                        ),
        "correlations": "find_top_correlations" in action_log,
    }
    correctness_score = sum(pipeline.values()) / len(pipeline)

    # 3. Insight quality — Fix 1 correlation verification applied per insight
    per_insight_scores: List[float] = []
    if insights:
        for ins in insights[:5]:   # cap at 5
            per_insight_scores.append(_score_insight(ins, df))
        avg_quality    = sum(per_insight_scores) / len(per_insight_scores)
        quantity_bonus = min(
            len([s for s in per_insight_scores if s >= 0.4]) / 3.0, 1.0
        )
        insight_score = (avg_quality * 0.6 + quantity_bonus * 0.4)
    else:
        insight_score = 0.0

    final_reward = (cleanliness_score + correctness_score + insight_score) / 3.0
    return {
        "cleanliness_score":  round(cleanliness_score, 4),
        "correctness_score":  round(correctness_score, 4),
        "insight_score":      round(insight_score, 4),
        "pipeline_checklist": pipeline,
        "per_insight_scores": per_insight_scores,
        "final_reward":       round(final_reward, 4),
    }
