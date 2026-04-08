"""
Medical Triage Environment - OpenEnv Implementation
Simulates real-world emergency department triage workflows.
"""

from __future__ import annotations
import json
import random
import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class TriageLevel(str, Enum):
    IMMEDIATE = "immediate"      # Red  - life-threatening, act now
    URGENT    = "urgent"         # Yellow - serious, can wait briefly
    DELAYED   = "delayed"        # Green  - stable, non-urgent
    EXPECTANT = "expectant"      # Black  - unsurvivable / minimal resources

VITALS_FIELDS = ["hr", "bp_sys", "bp_dia", "rr", "spo2", "temp", "gcs"]

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PatientRecord(BaseModel):
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    vitals: Dict[str, Any]
    history: List[str]
    symptoms: List[str]
    # Hidden from agent until ordered
    lab_results: Optional[Dict[str, Any]] = None
    imaging_results: Optional[Dict[str, Any]] = None
    true_triage_level: str          # ground truth - never shown directly
    true_diagnosis: str             # ground truth
    notes: List[str] = Field(default_factory=list)

class Observation(BaseModel):
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    vitals: Dict[str, Any]
    history: List[str]
    symptoms: List[str]
    available_actions: List[str]
    revealed_labs: Optional[Dict[str, Any]] = None
    revealed_imaging: Optional[Dict[str, Any]] = None
    agent_notes: List[str] = Field(default_factory=list)
    step_count: int = 0
    budget_remaining: int = 10

class Action(BaseModel):
    action_type: str   # "order_labs" | "order_imaging" | "assign_triage" | "add_note" | "consult"
    payload: Optional[Dict[str, Any]] = None

class Reward(BaseModel):
    value: float
    reason: str

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

# ---------------------------------------------------------------------------
# Patient Scenarios (Tasks: Easy / Medium / Hard)
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, List[PatientRecord]] = {

    # ── EASY ────────────────────────────────────────────────────────────────
    "easy_triage": [
        PatientRecord(
            patient_id="P001",
            age=28, sex="female",
            chief_complaint="Ankle pain after sports injury",
            vitals={"hr": 78, "bp_sys": 118, "bp_dia": 76, "rr": 14, "spo2": 99, "temp": 36.7, "gcs": 15},
            history=["No chronic illnesses", "No medications"],
            symptoms=["right ankle swelling", "tenderness on palpation", "able to bear weight"],
            lab_results={"WBC": 7.2, "HGB": 13.8},
            imaging_results={"xray_ankle": "No fracture, soft tissue swelling"},
            true_triage_level="delayed",
            true_diagnosis="Ankle sprain",
        ),
        PatientRecord(
            patient_id="P002",
            age=45, sex="male",
            chief_complaint="Severe chest pain radiating to left arm",
            vitals={"hr": 102, "bp_sys": 160, "bp_dia": 95, "rr": 22, "spo2": 94, "temp": 37.1, "gcs": 15},
            history=["Hypertension", "Smoker", "Family history of MI"],
            symptoms=["diaphoresis", "nausea", "crushing chest pain x 30 min"],
            lab_results={"troponin": 2.4, "CK_MB": 18.5, "WBC": 9.1},
            imaging_results={"ECG": "ST elevation in leads II, III, aVF"},
            true_triage_level="immediate",
            true_diagnosis="STEMI",
        ),
        PatientRecord(
            patient_id="P003",
            age=6, sex="male",
            chief_complaint="Fever and ear pain",
            vitals={"hr": 105, "bp_sys": 98, "bp_dia": 62, "rr": 22, "spo2": 98, "temp": 38.9, "gcs": 15},
            history=["No chronic illness", "Up to date vaccinations"],
            symptoms=["left ear pain", "fever 3 days", "irritability", "no neck stiffness"],
            lab_results={"WBC": 13.4, "CRP": 24},
            imaging_results={"xray_ear": "Not indicated / not performed"},
            true_triage_level="urgent",
            true_diagnosis="Acute otitis media",
        ),
    ],

    # ── MEDIUM ──────────────────────────────────────────────────────────────
    "medium_triage": [
        PatientRecord(
            patient_id="P004",
            age=67, sex="female",
            chief_complaint="Sudden onset severe headache — 'worst of life'",
            vitals={"hr": 88, "bp_sys": 178, "bp_dia": 104, "rr": 18, "spo2": 97, "temp": 37.0, "gcs": 14},
            history=["Hypertension", "No prior headaches"],
            symptoms=["thunderclap headache", "neck stiffness", "photophobia", "vomiting"],
            lab_results={"WBC": 10.2, "glucose": 118},
            imaging_results={"CT_head": "Hyperdense blood in basal cisterns"},
            true_triage_level="immediate",
            true_diagnosis="Subarachnoid hemorrhage",
        ),
        PatientRecord(
            patient_id="P005",
            age=52, sex="male",
            chief_complaint="Shortness of breath and leg swelling x 3 days",
            vitals={"hr": 112, "bp_sys": 128, "bp_dia": 82, "rr": 24, "spo2": 91, "temp": 37.4, "gcs": 15},
            history=["Recent long-haul flight 1 week ago", "Obese BMI 34"],
            symptoms=["right calf pain and swelling", "pleuritic chest pain", "hemoptysis"],
            lab_results={"D_dimer": 4800, "troponin": 0.08, "BNP": 320},
            imaging_results={"CT_pulmonary_angiogram": "Bilateral pulmonary emboli"},
            true_triage_level="immediate",
            true_diagnosis="Pulmonary embolism",
        ),
        PatientRecord(
            patient_id="P006",
            age=34, sex="female",
            chief_complaint="Abdominal pain and missed period",
            vitals={"hr": 116, "bp_sys": 90, "bp_dia": 58, "rr": 20, "spo2": 98, "temp": 36.9, "gcs": 15},
            history=["LMP 7 weeks ago", "IUD in situ"],
            symptoms=["right lower quadrant pain", "shoulder tip pain", "vaginal spotting"],
            lab_results={"beta_hCG": 1840, "HGB": 10.2},
            imaging_results={"pelvic_US": "No IUP seen, free fluid in Pouch of Douglas"},
            true_triage_level="immediate",
            true_diagnosis="Ectopic pregnancy with hemoperitoneum",
        ),
    ],

    # ── HARD ────────────────────────────────────────────────────────────────
    "hard_triage": [
        PatientRecord(
            patient_id="P007",
            age=72, sex="male",
            chief_complaint="Confusion and fever in nursing home resident",
            vitals={"hr": 122, "bp_sys": 86, "bp_dia": 52, "rr": 26, "spo2": 93, "temp": 39.4, "gcs": 11},
            history=["Dementia", "DM type 2", "CKD stage 3", "on metformin"],
            symptoms=["acute confusion", "dysuria reported by carer", "hypotension", "tachycardia"],
            lab_results={"WBC": 21.4, "lactate": 3.8, "creatinine": 2.9, "procalcitonin": 18.2, "urinalysis": "nitrites++, WBC++"},
            imaging_results={"CXR": "No infiltrates", "CT_abdomen": "Perinephric stranding right kidney"},
            true_triage_level="immediate",
            true_diagnosis="Urosepsis with septic shock",
        ),
        PatientRecord(
            patient_id="P008",
            age=19, sex="female",
            chief_complaint="Altered mental status and rash",
            vitals={"hr": 134, "bp_sys": 78, "bp_dia": 44, "rr": 30, "spo2": 95, "temp": 40.1, "gcs": 10},
            history=["No known illnesses", "University dormitory resident"],
            symptoms=["non-blanching petechial rash", "photophobia", "neck stiffness", "prodrome of URI 2 days ago"],
            lab_results={"WBC": 24.8, "CRP": 310, "CSF_WBC": 2800, "CSF_glucose": 1.2, "CSF_protein": 4.1},
            imaging_results={"CT_head": "No mass lesion"},
            true_triage_level="immediate",
            true_diagnosis="Bacterial meningitis / meningococcemia",
        ),
        PatientRecord(
            patient_id="P009",
            age=58, sex="male",
            chief_complaint="Chest pain and back pain — tearing quality",
            vitals={"hr": 96, "bp_sys": 168, "bp_sys_left": 112, "bp_dia": 90, "rr": 20, "spo2": 96, "temp": 37.2, "gcs": 15},
            history=["Hypertension poorly controlled", "Marfan features noted"],
            symptoms=["tearing chest/back pain radiating to abdomen", "blood pressure differential between arms > 50 mmHg", "new aortic regurgitation murmur"],
            lab_results={"D_dimer": 6200, "troponin": 0.04, "HGB": 12.4},
            imaging_results={"CT_aortogram": "Type A aortic dissection, involvement of coronary ostia"},
            true_triage_level="immediate",
            true_diagnosis="Type A aortic dissection",
        ),
    ],
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MedicalTriageEnv:
    """
    Medical Triage Environment.
    
    The agent is an ED triage clinician. Given a patient presentation,
    the agent must:
      1. Gather information (order labs / imaging / consult)
      2. Assign a triage level
      3. Document reasoning

    Each information-gathering action costs 1 budget unit.
    The agent has a max of 10 actions per episode.
    """

    TASK_NAMES = list(SCENARIOS.keys())
    ENV_NAME   = "medical-triage"

    def __init__(self, task_name: str = "easy_triage", seed: int = 42):
        if task_name not in SCENARIOS:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {self.TASK_NAMES}")
        self.task_name = task_name
        self.seed = seed
        self._rng = random.Random(seed)
        self._patients: List[PatientRecord] = []
        self._current_patient: Optional[PatientRecord] = None
        self._obs: Optional[Observation] = None
        self._step_count = 0
        self._budget = 10
        self._done = False
        self._revealed_labs = None
        self._revealed_imaging = None
        self._agent_notes: List[str] = []
        self._triage_assigned: Optional[str] = None
        self._episode_rewards: List[float] = []
        self._patient_index = 0

    # ── OpenEnv Interface ───────────────────────────────────────────────────

    def reset(self) -> Observation:
        scenarios = SCENARIOS[self.task_name]
        self._current_patient = scenarios[self._patient_index % len(scenarios)]
        self._patient_index += 1
        self._step_count = 0
        self._budget = 10
        self._done = False
        self._revealed_labs = None
        self._revealed_imaging = None
        self._agent_notes = []
        self._triage_assigned = None
        self._episode_rewards = []
        self._obs = self._build_obs()
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self._done:
            return self._obs, 0.0, True, {"error": "Episode already done"}

        reward, done, info = self._process_action(action)
        self._step_count += 1
        self._episode_rewards.append(reward)
        self._obs = self._build_obs()
        self._done = done
        return self._obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        if self._current_patient is None:
            return {}
        return {
            "task": self.task_name,
            "patient_id": self._current_patient.patient_id,
            "step_count": self._step_count,
            "budget_remaining": self._budget,
            "triage_assigned": self._triage_assigned,
            "done": self._done,
            "cumulative_reward": round(sum(self._episode_rewards), 4),
        }

    def close(self):
        self._done = True

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_obs(self) -> Observation:
        p = self._current_patient
        available = self._available_actions()
        return Observation(
            patient_id=p.patient_id,
            age=p.age,
            sex=p.sex,
            chief_complaint=p.chief_complaint,
            vitals=p.vitals,
            history=p.history,
            symptoms=p.symptoms,
            available_actions=available,
            revealed_labs=self._revealed_labs,
            revealed_imaging=self._revealed_imaging,
            agent_notes=list(self._agent_notes),
            step_count=self._step_count,
            budget_remaining=self._budget,
        )

    def _available_actions(self) -> List[str]:
        actions = []
        if self._budget > 0 and not self._done:
            if self._revealed_labs is None:
                actions.append("order_labs")
            if self._revealed_imaging is None:
                actions.append("order_imaging")
            actions.append("add_note")
            actions.append("consult")
        if not self._done:
            actions.append("assign_triage:{immediate|urgent|delayed|expectant}")
        return actions

    def _process_action(self, action: Action) -> Tuple[float, bool, Dict]:
        p = self._current_patient
        atype = action.action_type.lower()

        # ── Budget exhausted ───────────────────────────────────────────────
        if self._budget <= 0 and atype not in ("assign_triage",):
            return -0.1, False, {"error": "Budget exhausted"}

        # ── Order labs ────────────────────────────────────────────────────
        if atype == "order_labs":
            if self._revealed_labs is not None:
                return -0.05, False, {"error": "Labs already ordered"}
            self._revealed_labs = p.lab_results or {}
            self._budget -= 1
            reward = 0.1 if p.lab_results else 0.0
            return reward, False, {"info": "Labs revealed"}

        # ── Order imaging ─────────────────────────────────────────────────
        if atype == "order_imaging":
            if self._revealed_imaging is not None:
                return -0.05, False, {"error": "Imaging already ordered"}
            self._revealed_imaging = p.imaging_results or {}
            self._budget -= 1
            reward = 0.1 if p.imaging_results else 0.0
            return reward, False, {"info": "Imaging revealed"}

        # ── Consult (costs budget, small reward) ──────────────────────────
        if atype == "consult":
            self._budget -= 1
            specialty = (action.payload or {}).get("specialty", "general")
            self._agent_notes.append(f"Consult: {specialty}")
            return 0.05, False, {"info": f"Consulted {specialty}"}

        # ── Add clinical note ─────────────────────────────────────────────
        if atype == "add_note":
            self._budget -= 1
            note = (action.payload or {}).get("text", "")
            self._agent_notes.append(note)
            # Small reward for noting key terms
            keywords = {"sepsis", "stemi", "embolism", "hemorrhage", "dissection",
                        "meningitis", "ectopic", "triage", "critical"}
            hit = any(kw in note.lower() for kw in keywords)
            return 0.05 if hit else 0.0, False, {"info": "Note added"}

        # ── Assign triage level ───────────────────────────────────────────
        if atype == "assign_triage":
            level = (action.payload or {}).get("level", "").lower()
            if level not in [e.value for e in TriageLevel]:
                return -0.2, False, {"error": f"Invalid triage level: {level}"}

            self._triage_assigned = level
            reward = self._score_triage(level, p)
            return reward, True, {"triage_assigned": level, "correct": p.true_triage_level == level}

        return -0.1, False, {"error": f"Unknown action: {atype}"}

    def _score_triage(self, assigned: str, p: PatientRecord) -> float:
        """
        Reward structure:
          - Correct level:                     +1.0
          - One level off (e.g. urgent vs immediate): +0.4
          - Wrong direction (delayed when immediate): -0.5
          - Used relevant information:         +0.1 bonus
        """
        true = p.true_triage_level
        level_order = [TriageLevel.EXPECTANT.value, TriageLevel.DELAYED.value,
                       TriageLevel.URGENT.value, TriageLevel.IMMEDIATE.value]

        if assigned == true:
            base = 1.0
        else:
            try:
                dist = abs(level_order.index(assigned) - level_order.index(true))
                if dist == 1:
                    base = 0.4
                else:
                    base = -0.5 if (true == "immediate" and assigned in ("delayed", "expectant")) else 0.0
            except ValueError:
                base = -0.2

        # Bonus for gathering information before deciding
        info_bonus = 0.0
        if self._revealed_labs is not None:
            info_bonus += 0.05
        if self._revealed_imaging is not None:
            info_bonus += 0.05

        return round(min(1.0, base + info_bonus), 4)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_episode(env: MedicalTriageEnv) -> float:
    """
    Returns a score in [0, 1] based on:
    - Triage accuracy (70%)
    - Information gathering (20%)
    - Efficiency (10%)
    """
    p = env._current_patient
    if p is None or env._triage_assigned is None:
        return 0.0

    # Triage accuracy
    correct = env._triage_assigned == p.true_triage_level
    triage_score = 1.0 if correct else 0.3

    # Information gathering
    info_score = 0.0
    if env._revealed_labs is not None:
        info_score += 0.5
    if env._revealed_imaging is not None:
        info_score += 0.5

    # Efficiency: fewer steps used = better
    steps_used = env._step_count
    efficiency = max(0.0, 1.0 - (steps_used / 10.0))

    final = 0.7 * triage_score + 0.2 * info_score + 0.1 * efficiency
    return round(final, 4)