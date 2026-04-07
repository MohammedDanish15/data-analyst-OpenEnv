# 🛒 Data Analyst RL Environment

An **OpenEnv-compatible reinforcement learning environment** that simulates a junior data analyst
working on real e-commerce datasets. Agents must explore, clean, transform, and extract structured
insights from messy sales data to earn rewards.

---

## 🎯 Why This Environment Exists

| Use-case | How this env helps |
|---|---|
| **Train AI analyst assistants** | Agents learn a full analytical workflow step-by-step with shaped rewards |
| **Automate dataset cleaning** | Reward signal drives agents toward zero missing/outlier data |
| **Prepare data for ML pipelines** | Trained agents produce clean, encoded, normalised DataFrames |
| **Benchmark agent reasoning** | 3 difficulty tiers with deterministic, anti-gaming graders |
| **Simulate enterprise workflows** | Mirrors real BI/analytics workflows: exploration → cleaning → insights |

---

## 📦 Project Structure

```
data-analyst-env/
├── Dockerfile
├── openenv.yaml
├── inference.py          # Baseline agent (OpenAI client → HF router)
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py
    ├── main.py           # FastAPI server — /reset /step /state /tasks /health
    ├── environment.py    # Core env: reset() / step() / state()
    ├── models.py         # Pydantic typed models
    ├── datasets.py       # Seeded dataset generators (reproducible)
    ├── graders.py        # Deterministic, anti-gaming graders
    └── tasks/
        └── __init__.py
```

---

## 🔭 Observation Space

```json
{
  "dataset_metadata": {
    "total_rows": 200,
    "total_columns": 5,
    "missing_counts": {"age": 30, "total_price": 20},
    "numeric_summary": {"age": {"mean": 43.1, "std": 14.2, "min": 18.0, "max": 69.0}},
    "column_types": {"age": "float64", "country": "object"}
  },
  "task_stage": "cleaning",
  "environment_health": {
    "step_number": 3,
    "invalid_action_count": 0,
    "data_quality_score": 0.875
  },
  "agent_progress": {
    "progress_percentage": 12.0,
    "intermediate_score": 0.35
  },
  "last_action_result": {"cells_filled": 30, "method": "mean"},
  "task_id": "task1",
  "task_description": "Explore the dataset and fill missing values…"
}
```

---

## ⚡ Action Space

| Category | Action | Params | Step Reward |
|---|---|---|---|
| Exploration | `view_head` | — | 0.05 first / 0.01 repeat |
| Exploration | `view_statistics` | — | 0.05 / 0.01 |
| Exploration | `view_column_types` | — | 0.05 / 0.01 |
| Exploration | `view_correlations` | — | 0.05 / 0.01 |
| Cleaning | `fill_missing_mean` | `{"column": "age"}` or omit for all | 0.15 if effective / 0.00 noop |
| Cleaning | `fill_missing_median` | `{"column": "age"}` | 0.15 / 0.00 |
| Cleaning | `drop_missing_rows` | — | 0.10 |
| Cleaning | `detect_outliers` | — | 0.05 |
| Cleaning | `remove_outliers` | `{"column": "age"}` or omit for all | 0.15 if effective |
| Transform | `normalize` | — | 0.10 |
| Transform | `standardize` | — | 0.10 |
| Transform | `encode_categorical` | — | 0.10 |
| Transform | `apply_minmax_scaling` | — | 0.10 |
| Analysis | `find_top_correlations` | — | 0.10 |
| Analysis | `generate_summary` | — | 0.10 |
| Analysis | `generate_insights` | `{"insights": [...]}` | 0.20 |
| Completion | `submit_report` | `{"insights": [...]}` optional | **Final grade 0–1** |
| Completion | `end_episode` | — | 0.00 |

> **Repeat penalty:** Calling the same action twice in a row halves the reward (`× 0.5`).
> `submit_report` and `end_episode` are exempt.

---

## 📊 Tasks

### Task 1 — Easy: Missing Value Fixing
- **Dataset:** 200 rows × 5 cols (`customer_id`, `age`, `total_price`, `quantity`, `country`)
- **Problems:** ~15% missing `age`, ~10% missing `total_price`
- **Goal:** Fill missing values with mean/median, submit
- **Grader:** cleanliness (nulls remaining) + correctness (fill action used) + insight (explored AND cleanliness > 0.95)
- **Verified score (optimal agent):** `1.000`

### Task 2 — Medium: Full Cleaning + Transformation
- **Dataset:** 500 rows × 8 cols
- **Problems:** Missing values + outliers (age=200, qty=-5) + inconsistent gender labels + bad categories (`???`)
- **Goal:** Fix all issues, encode categoricals, normalise, submit
- **Grader:** cleanliness + **outcome-verified** encoding (checks `dtype`, not just action log) + transform verified via column range
- **Verified score (optimal agent):** `0.942`

### Task 3 — Hard: Full Analysis + Insight Generation
- **Dataset:** 1000 rows × 12 cols with **hidden correlations**:
  - High `discount` → higher `return_flag` rate
  - `Electronics` → higher `unit_price` AND ~45% return rate
  - `customer_lifetime_value` correlates with `quantity × unit_price`
- **Goal:** Complete pipeline + find correlations + submit ≥3 structured insights
- **Insight format:**
  ```json
  {"finding": "Electronics has highest return rate", "metric": "return_rate", "value": 0.45}
  ```
- **Grader:** cleanliness + pipeline checklist (outcome-verified) + **correlation verification** (claimed values checked against actual DataFrame within ±0.20 tolerance)
- **Verified score (optimal agent):** `0.940`

---

## 🏆 Reward Design

```
Step reward      ∈ [0.00, 0.20]  per action based on usefulness
Repeat penalty   × 0.5           if same action repeated consecutively
Final reward     ∈ [0.00, 1.00]  from submit_report grader:
                                   = (cleanliness + correctness + insight) / 3
```

### Anti-Gaming Properties
| Exploit | Defence |
|---|---|
| Submit without cleaning | `insight_score` requires `cleanliness > 0.95` |
| Call encode but skip it | Grader checks `df[col].dtype`, not `action_log` |
| Fabricate correlation values | Claimed value verified against `df[col1].corr(df[col2])` ±0.20 |
| Spam same action | Consecutive repeat → `reward × 0.5` |
| Loop invalid actions | ≥5 invalid actions terminates episode |

---

## 🚀 Setup & Usage

### Docker (recommended)

```bash
docker build -t data-analyst-env .
docker run -p 7860:7860 data-analyst-env
```

### Local

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset?task_id=task1` | Start/reset episode |
| `POST` | `/step?task_id=task1` | Execute action |
| `GET` | `/state?task_id=task1` | Current state |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/health` | Health check |

### Example

```bash
# Start episode
curl -X POST http://localhost:7860/reset?task_id=task1

# Explore
curl -X POST http://localhost:7860/step?task_id=task1 \
  -H "Content-Type: application/json" \
  -d '{"action_type": "view_statistics", "params": {}}'

# Fill missing values
curl -X POST http://localhost:7860/step?task_id=task1 \
  -H "Content-Type: application/json" \
  -d '{"action_type": "fill_missing_mean", "params": {"column": "age"}}'

# Submit report
curl -X POST http://localhost:7860/step?task_id=task1 \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_report", "params": {}}'
```

---

## 🤖 Running the Baseline Agent

```bash
export HF_TOKEN="your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Baseline Scores (Qwen/Qwen2.5-72B-Instruct)

| Task | Score | Success (≥0.6) |
|---|---|---|
| task1 — easy   | ~0.72 | ✓ |
| task2 — medium | ~0.65 | ✓ |
| task3 — hard   | ~0.55 | ✗ |
| **Optimal agent** | **0.961 mean** | — |

---

## 📋 Environment Variables

| Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | — | ✓ |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | — |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | — |
| `ENV_URL` | `http://localhost:7860` | — |

---

## 🧠 Episode Termination

| Condition | Trigger |
|---|---|
| Agent submits report | `submit_report` action |
| Max steps exceeded | `step_number >= 25` |
| Too many invalid actions | `invalid_action_count >= 5` |
| Agent ends early | `end_episode` action (reward = 0) |

---

## ✅ Pre-Submission Checklist

```bash
# 1. Run openenv validator
pip install openenv-core
openenv validate

# 2. Build Docker image
docker build -t data-analyst-env .
docker run -p 7860:7860 data-analyst-env

# 3. Ping health
curl http://localhost:7860/health

# 4. Run baseline
export HF_TOKEN=...
python inference.py
```

---

## 📝 Changelog

### v1.1 (current)
- **Fix 1:** Task 3 — correlation values in insights verified against actual DataFrame (±0.20 tolerance). Fabricated values penalised at `× 0.3`.
- **Fix 2:** Task 1 — insight submit bonus now requires `cleanliness_score > 0.95`. Eliminates free points for submitting uncleaned data.
- **Fix 3:** Task 2 — encoding and transform correctness checks use actual DataFrame dtype, not action log presence.
- **Fix 4:** Repeated consecutive actions penalised at `reward × 0.5`.
- **Fix 5:** Success threshold raised from `0.5` → `0.6`.

### v1.0
- Initial release. Three tasks, 18 actions, seeded datasets.
