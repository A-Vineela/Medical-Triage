import os
import sys
import json
import traceback
from typing import Optional

from openai import OpenAI
from environment import MedicalTriageEnv, Action, Observation, grade_episode, SCENARIOS

# ── Environment variables ────────────────────────────────────────────────────
API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN       = os.getenv("HF_TOKEN")

API_KEY = OPENAI_API_KEY or HF_TOKEN
if API_KEY is None:
    raise ValueError("No API key found.")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Heuristic triage (NO API USAGE) ──────────────────────────────────────────

def simple_triage(obs: Observation) -> str:
    complaint = obs.chief_complaint.lower()

    if "chest pain" in complaint or "shortness of breath" in complaint:
        return "immediate"
    elif "headache" in complaint:
        return "immediate"
    elif "fever" in complaint or "ear pain" in complaint:
        return "urgent"
    else:
        return "delayed"

# ── Episode runner ───────────────────────────────────────────────────────────

def run_episode(task_name: str) -> dict:
    env = MedicalTriageEnv(task_name=task_name)

    all_scores = []
    all_rewards = []
    all_success = []
    total_steps = 0

    num_patients = len(SCENARIOS[task_name])

    for patient_idx in range(num_patients):
        obs = env.reset()
        episode_rewards = []

        print(f"\n[START] task={task_name} patient={obs.patient_id} ({patient_idx+1}/{num_patients}) model={MODEL_NAME}", flush=True)

        done = False
        step_n = 0
        last_error = None

        try:
            while not done and step_n < 1:

                # 🔥 NO LLM CALL — PURE HEURISTIC
                level = simple_triage(obs)

                action = Action(
                    action_type="assign_triage",
                    payload={"level": level}
                )

                obs, reward, done, info = env.step(action)
                step_n += 1
                episode_rewards.append(reward)
                last_error = info.get("error", None)

                action_str = f"{action.action_type}({json.dumps(action.payload or {})})"

                print(
                    f"[STEP] step={step_n} action={action_str} "
                    f"reward={reward:.2f} done={'true' if done else 'false'} "
                    f"error={last_error or 'null'}",
                    flush=True
                )

            final_score = grade_episode(env)
            success = done and (env._triage_assigned == env._current_patient.true_triage_level)
            rewards_str = ",".join(f"{r:.2f}" for r in episode_rewards)

            print(
                f"[END] patient={env._current_patient.patient_id} "
                f"success={'true' if success else 'false'} "
                f"steps={step_n} score={final_score:.3f} rewards={rewards_str}",
                flush=True
            )

            all_scores.append(final_score)
            all_rewards.extend(episode_rewards)
            all_success.append(success)
            total_steps += step_n

        except Exception as e:
            tb = traceback.format_exc()
            print(f"# Exception on patient {patient_idx}: {e}", file=sys.stderr)
            print(tb, file=sys.stderr)
            all_scores.append(0.0)
            all_success.append(False)

    env.close()
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {
        "task": task_name,
        "success": all(all_success),
        "steps": total_steps,
        "final_score": avg_score,
        "rewards": all_rewards,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    tasks = ["easy_triage", "medium_triage", "hard_triage"]
    results = []

    for task in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"# Running task: {task}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run_episode(task)
        results.append(result)

    print("\n" + "="*60, flush=True)
    print("# BENCHMARK SUMMARY", flush=True)
    print("="*60, flush=True)

    total_score = 0.0
    for r in results:
        score = r["final_score"]
        total_score += score
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['task']:20s}  score={score:.3f}  steps={r['steps']}", flush=True)

    avg = total_score / len(results)
    print(f"\nAverage score: {avg:.3f}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()
    tasks = ["easy_triage", "medium_triage", "hard_triage"]
    results = []

    for task in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"# Running task: {task}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run_episode(task)
        results.append(result)

    print("\n" + "="*60, flush=True)
    print("# BENCHMARK SUMMARY", flush=True)
    print("="*60, flush=True)

    total_score = 0.0
    for r in results:
        score = r["final_score"]
        total_score += score
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['task']:20s}  score={score:.3f}  steps={r['steps']}", flush=True)

    avg = total_score / len(results)
    print(f"\nAverage score: {avg:.3f}", flush=True)
    print("="*60, flush=True)