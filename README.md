---
title: Medical Triage
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🏥 Medical Triage Environment

> An OpenEnv RL benchmark simulating real-world emergency department triage workflows.

---

## Overview & Motivation

Emergency Department (ED) triage is a high-stakes, time-critical decision-making task performed by trained clinicians. The goal is to correctly prioritize patients by severity so that those who are most critical receive care first.

This environment simulates that process: the agent acts as a triage clinician, is presented with a patient's vitals, history, and symptoms, and must decide when to gather additional information (labs, imaging) versus when to act and assign a triage priority level.

The environment tests:
- Clinical pattern recognition
- Appropriate use of investigations
- Reasoning under resource constraints (budget)
- Correct urgency classification

---

## Action Space

| Action | Payload | Cost | Effect |
|--------|---------|------|--------|
| `order_labs` | `{}` | 1 budget | Reveals lab results |
| `order_imaging` | `{}` | 1 budget | Reveals imaging results |
| `add_note` | `{"text": "..."}` | 1 budget | Documents clinical reasoning |
| `consult` | `{"specialty": "..."}` | 1 budget | Records specialist consult |
| `assign_triage` | `{"level": "immediate\|urgent\|delayed\|expectant"}` | 0 | Ends episode, triggers scoring |

### Triage Levels
| Level | Color | Meaning |
|-------|-------|---------|
| `immediate` | 🔴 Red | Life-threatening, intervene NOW |
| `urgent` | 🟡 Yellow | Serious, can wait 15–30 min |
| `delayed` | 🟢 Green | Stable, non-urgent |
| `expectant` | ⚫ Black | Unsurvivable / minimal resources |

---

## Observation Space

```json
{
  "patient_id": "P001",
  "age": 45,
  "sex": "male",
  "chief_complaint": "Severe chest pain radiating to left arm",
  "vitals": {
    "hr": 102, "bp_sys": 160, "bp_dia": 95,
    "rr": 22, "spo2": 94, "temp": 37.1, "gcs": 15
  },
  "history": ["Hypertension", "Smoker"],
  "symptoms": ["diaphoresis", "nausea", "crushing chest pain x 30 min"],
  "available_actions": ["order_labs", "order_imaging", "assign_triage:..."],
  "revealed_labs": null,
  "revealed_imaging": null,
  "agent_notes": [],
  "step_count": 0,
  "budget_remaining": 10
}
```

---

## Tasks

### 🟢 Easy — `easy_triage`
**3 patients with classic, unambiguous presentations.**

| Patient | Presentation | True Triage | Expected Difficulty |
|---------|-------------|-------------|---------------------|
| P001 | Ankle injury after sport | delayed | Easy |
| P002 | Crushing chest pain + diaphoresis | immediate | Easy |
| P003 | Child with fever + ear pain | urgent | Easy |

Baseline score: ~0.82

---

### 🟡 Medium — `medium_triage`
**3 patients with serious multi-system emergencies requiring investigation.**

| Patient | Presentation | True Triage | Expected Difficulty |
|---------|-------------|-------------|---------------------|
| P004 | "Worst headache of life" + neck stiffness | immediate | Medium |
| P005 | SOB + leg swelling post long-haul flight | immediate | Medium |
| P006 | Abdo pain + missed period + hypotension | immediate | Medium |

Baseline score: ~0.65

---

### 🔴 Hard — `hard_triage`
**3 critically ill patients with complex, overlapping presentations.**

| Patient | Presentation | True Triage | Expected Difficulty |
|---------|-------------|-------------|---------------------|
| P007 | Confused nursing home resident, hypotension | immediate | Hard |
| P008 | Young patient with rash, neck stiffness, shock | immediate | Hard |
| P009 | Tearing chest pain, BP differential between arms | immediate | Hard |

Baseline score: ~0.50

---

## Reward Function

Rewards are designed to be **incremental**, not just end-of-episode:

| Event | Reward |
|-------|--------|
| Order labs (with results available) | +0.10 |
| Order imaging (with results available) | +0.10 |
| Add note with clinical keywords | +0.05 |
| Consult specialty | +0.05 |
| Correct triage level | +1.00 |
| One level off | +0.40 |
| Two levels off | 0.00 |
| Dangerous under-triage (immediate → delayed/expectant) | −0.50 |
| Duplicate action | −0.05 |
| Budget exhausted, non-assignment action | −0.10 |

### Episode Grade Formula
```
score = 0.70 * triage_accuracy
      + 0.20 * information_gathering  
      + 0.10 * efficiency
```

---

## Setup & Usage

### Local Development

```bash
git clone <your-repo>
cd medical-triage-env

pip install -r requirements.txt

# Option A: OpenEnv spec standard key (works with any OpenAI-compatible endpoint)
export OPENAI_API_KEY=your_key_here
# Option B: HuggingFace deployment key
export HF_TOKEN=your_hf_token_here

export API_BASE_URL=https://router.huggingface.co/v1   # default
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct            # default

python inference.py
```

### Docker

```bash
# Build
docker build -t medical-triage-env .

# Run — pass HF_TOKEN from your shell, never hardcode it
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  medical-triage-env
```

### OpenEnv Validation

```bash
openenv validate
```

---

## Baseline Performance Scores

Evaluated with `gpt-4.1-mini` at temperature 0.2:

| Task | Score | Success Rate |
|------|-------|-------------|
| easy_triage | 0.82 | 90% |
| medium_triage | 0.65 | 65% |
| hard_triage | 0.50 | 48% |
| **Average** | **0.66** | **68%** |

---

## Project Structure

```
medical-triage-env/
├── inference.py        # Main OpenEnv submission script (entry point)
├── environment.py      # MedicalTriageEnv implementation
├── openenv.yaml        # OpenEnv metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
└── README.md           # This file
```

---

## Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | — | ✅ Yes (OpenEnv spec standard) OR use HF_TOKEN |
| `HF_TOKEN` | — | ✅ Yes (HuggingFace deployment) OR use OPENAI_API_KEY |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No |

---

## Tags

`openenv` `medical` `triage` `healthcare` `real-world` `rl-benchmark`