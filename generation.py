"""
NavIO — generation.py
GenAI generation: Medical Necessity Justification + Next-Session Strategy.

Ported from Part 4 (navio_part4_generation.py). Only change from the Colab
version: HF_TOKEN comes from the Space's environment (a Repository secret)
instead of `google.colab.userdata`. All prompt logic, few-shot examples, and
the hallucination-guard rules are unchanged from the validated Part 4 script.

Model: meta-llama/Llama-3.1-8B-Instruct via the HF Inference API — an
open-weight model, just hosted remotely; this is "using a Hugging Face
model," not a closed-API MaaS dependency.

Two separate calls per session, at different temperatures: the Justification
(low temperature, factual, insurance-facing) and the Strategy (moderate
temperature, a clinical recommendation that benefits from some flexibility).
"""

import os
import time
import random
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
GENERATION_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
client = InferenceClient(model=GENERATION_MODEL_ID, token=HF_TOKEN, timeout=120)

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 3


# --- Retry wrapper (synchronous — a live, one-session-at-a-time flow) -------
class QuotaExceededError(Exception):
    """HTTP 402 — non-transient, don't retry."""


def _is_payment_required(e):
    status_code = getattr(getattr(e, "response", None), "status_code", None)
    return status_code == 402 or ("402" in str(e) and "payment" in str(e).lower())


def call_model_with_retry_sync(messages, max_tokens, temperature):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if _is_payment_required(e):
                raise QuotaExceededError(str(e)) from e
            last_error = e
            wait_time = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            print(f"Attempt {attempt}/{MAX_RETRIES} failed ({e.__class__.__name__}). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    raise last_error


# --- Diagnosis-driven grounding reference + prompts -------------------------
DIAGNOSIS_MODALITY_HINTS = {
    "MDD": "behavioral activation, cognitive restructuring of negative thought patterns, monitoring anhedonia/sleep/appetite changes",
    "GAD": "worry-focused CBT, relaxation and grounding techniques, addressing intolerance of uncertainty",
    "PTSD": "trauma-focused approaches (e.g. cognitive processing therapy principles), grounding techniques for triggers, careful pacing around trauma processing",
    "Substance_Use_Disorder": "motivational interviewing, relapse-prevention planning, identifying triggers and cravings",
    "Grief_Bereavement": "grief-focused processing, meaning reconstruction, monitoring for complicated grief symptoms",
    "Panic_Disorder": "panic-focused CBT, interoceptive exposure principles, psychoeducation on panic physiology",
    "Social_Anxiety": "graded exposure to social situations, cognitive restructuring of social fears, behavioral experiments",
    "Bipolar_Disorder": "mood stabilization focus, sleep/routine regulation, early-warning-sign monitoring, psychoeducation",
}

JUSTIFICATION_SYSTEM_PROMPT = """You are a clinical documentation assistant helping a licensed \
mental health professional draft a Medical Necessity Justification for an insurance claim, \
following CMS/APA-aligned behavioral health documentation standards.

Every justification must, through natural clinical narrative (never as a literal checklist or by \
naming these items explicitly), reflect all of the following:
1. The specific DSM-5/ICD-10 diagnosis being treated (not solely life coaching or personal growth).
2. Concrete functional impairment — behavioral, not vague (e.g. missed work deadlines, withdrawal \
from relationships, self-care deficits), never just "feels bad".
3. The evidence-based treatment approach actually used in session, appropriate to that diagnosis.
4. Clear clinical necessity — this is not for mild stress or lifestyle enhancement.
5. An expectation that the intervention will improve symptoms, prevent deterioration, or stabilize \
a chronic condition.
6. That the level of care billed (standard vs. crisis) is the least restrictive one appropriate to \
the presentation.
7. Where applicable, the patient's ongoing treatment trajectory and progress monitoring.

Additional rules:
8. Write in formal, professional clinical English suitable for direct submission to an insurer.
9. Base every claim STRICTLY on the session data provided — do not invent symptoms, history, or \
details not given.
10. You may ONLY describe the patient's clinical state using the specific fields provided \
(Primary_Diagnosis, Risk_Score/Risk_Level, Crisis_Flag, Progress, and what is explicitly stated in \
the transcript). Do NOT invent or infer additional clinical constructs that were not given to you — \
for example, do not describe "independence," "self-sufficiency," "insight," "motivation," or similar \
psychological qualities unless the transcript or session data explicitly supports them. If you need \
to narrate deterioration or improvement, ground it directly in the Risk_Score/Progress values given, \
not in a new trait you've invented to explain them.
11. Explicitly justify the assigned CPT code using standard CPT/insurance language, following the \
CPT-specific guidance provided for this session.
12. This is a DRAFT for the treating clinician to review and finalize, not a final document — do \
not state or imply it has already been reviewed or approved.
13. One tightly-written paragraph, 110-170 words. No headers, no bullet points, no commentary \
outside the paragraph.

Two example Justifications follow (one standard, one crisis) demonstrating the expected structure, \
tone, and terminology. Match their professional register and level of specificity, but every fact \
in your output must come from the NEW session's actual data — never reuse details from these examples.
"""

STRATEGY_SYSTEM_PROMPT = """You are a clinical documentation assistant helping a licensed \
mental health professional plan their approach to an upcoming session.

Rules:
1. Recommendations must be clearly tailored to the patient's Primary_Diagnosis — a strategy for \
PTSD must look substantively different from one for Social Anxiety Disorder or MDD. Use the \
"Typical evidence-based focus areas" reference provided, adapted to this specific session's \
metrics — do not apply it generically or recite it verbatim.
2. Explicitly account for the current Risk_Level, Crisis_Flag, Dynamic engagement level, and \
Progress label. If Progress is "Mixed Signal", explicitly acknowledge the conflicting indicators \
(e.g. improving risk but more defensive engagement, or vice versa) and recommend a cautious, \
exploratory approach rather than assuming a single clear direction.
3. If similar past cases are provided, you may reference the general pattern they suggest, but \
do not claim to know outcomes for THIS patient with certainty.
4. Use clinically appropriate hedging language ("consider", "may benefit from") rather than \
directive commands — this is a suggestion for clinical judgment, not an instruction to follow \
blindly.
5. Describe the patient's clinical state ONLY using the fields provided (Primary_Diagnosis, \
Risk_Level, Crisis_Flag, Dynamic, Progress). Do NOT invent additional psychological traits (e.g. \
"independence," "insight," "motivation") to explain a metric — reference the metric itself \
(e.g. "the Defensive engagement style" or "the elevated Risk_Score"), not a new construct you've \
made up to narrate it.
6. One paragraph, 100-160 words. No headers, no bullet points, no commentary outside the paragraph.
"""


# --- CPT differentiation guidance (standard vs. crisis need different signals emphasized) --
def build_cpt_guidance(crisis_flag, duration_minutes):
    if crisis_flag:
        return (
            f"This session is being billed as CRISIS psychotherapy (CPT 90839, first 60 minutes). "
            f"The justification MUST establish imminent danger or acute decompensation (e.g. active "
            f"suicidal/self-harm risk, severe acute distress, psychotic agitation) and describe the "
            f"complex, urgent intervention performed (risk assessment, safety planning, de-escalation, "
            f"and/or coordination with family or crisis services). Session duration recorded: "
            f"{duration_minutes} minutes (valid crisis range: 30-74 minutes)."
        )
    return (
        f"This session is being billed as STANDARD outpatient psychotherapy. The justification MUST "
        f"connect the patient's active symptoms and functional impairment to the specific evidence-based "
        f"treatment techniques used in session, tied to their diagnosis and treatment plan. Session "
        f"duration recorded: {duration_minutes} minutes (standard 45-minute session validity range: "
        f"38-52 minutes)."
    )


# --- Few-shot exemplars: grounded in CMS/APA-documented conventions, not our own synthetic ---
# data — inserted as real prior turns, not text inside the system prompt, since
# instruction-tuned chat models follow demonstrated input->output turns more reliably
# than examples embedded as static text in the system message.
FEWSHOT_STANDARD_INPUT = """Session data:
- Primary_Diagnosis: GAD
- Risk_Score: 4.5 (Medium)
- Crisis_Flag: False
- Session_Duration: 45 minutes
- Assigned CPT Code: 90834

This session is being billed as STANDARD outpatient psychotherapy. The justification MUST connect \
the patient's active symptoms and functional impairment to the specific evidence-based treatment \
techniques used in session, tied to their diagnosis and treatment plan. Session duration recorded: \
45 minutes (standard 45-minute session validity range: 38-52 minutes).

Session transcript:
[00:00] Dr. Carter: How has your anxiety been since we last spoke?
[00:10] Patient: Honestly, still pretty constant. I've been having trouble focusing at work and I missed two deadlines this month.
[00:25] Dr. Carter: Let's look at the thought that comes up right before you feel that panic about deadlines.
[00:40] Patient: I keep thinking I'm going to get fired, even though nothing's actually happened.
[00:50] Dr. Carter: That's a good example of catastrophic thinking. Let's practice reframing it, and I'd like you to try approaching one of the social events you've been avoiding this week.

Write the Medical Necessity Justification paragraph for this session."""

FEWSHOT_STANDARD_OUTPUT = """Patient presents with Generalized Anxiety Disorder (ICD-10 F41.1), \
reporting persistent worry, muscle tension, and sleep-onset difficulty of more than six months' \
duration. Functional impairment is evidenced by reported difficulty concentrating at work, resulting \
in two missed deadlines this month, and withdrawal from previously enjoyed social activities. During \
this 45-minute session, the clinician utilized cognitive restructuring to address catastrophic \
thinking patterns and assigned a graded exposure exercise targeting a specific avoided social \
situation. The patient demonstrated increased insight but continues to exhibit clinically significant \
anxiety symptoms warranting continued outpatient psychotherapy. CPT code 90834 is medically necessary \
given the moderate symptom severity and the clinical focus required within a standard-length session."""

FEWSHOT_CRISIS_INPUT = """Session data:
- Primary_Diagnosis: MDD
- Risk_Score: 9.0 (High)
- Crisis_Flag: True
- Session_Duration: 60 minutes
- Assigned CPT Code: 90839

This session is being billed as CRISIS psychotherapy (CPT 90839, first 60 minutes). The \
justification MUST establish imminent danger or acute decompensation (e.g. active suicidal/self-harm \
risk, severe acute distress, psychotic agitation) and describe the complex, urgent intervention \
performed (risk assessment, safety planning, de-escalation, and/or coordination with family or \
crisis services). Session duration recorded: 60 minutes (valid crisis range: 30-74 minutes).

Session transcript:
[00:00] Dr. Carter: I'm glad you called. Can you tell me what's going on right now?
[00:05] Patient: (crying) I don't think I can keep going. I have the pills, I've thought about how I'd do it.
[00:15] Dr. Carter: Thank you for telling me. I want to keep you safe right now — can we call your sister together to help secure those?
[00:30] Patient: Okay... okay, I can do that.
[00:45] Dr. Carter: Let's also build a safety plan for tonight and tomorrow before you leave.

Write the Medical Necessity Justification paragraph for this session."""

FEWSHOT_CRISIS_OUTPUT = """Patient, diagnosed with Major Depressive Disorder, Recurrent, Severe \
(ICD-10 F33.2), contacted the office requesting an urgent same-day session following a significant \
personal loss, presenting in acute distress with tearfulness and psychomotor agitation. Urgent \
psychosocial assessment and mental status exam revealed active suicidal ideation with a specific \
plan; intent was assessed as fluctuating. A collaborative safety plan was developed, means \
restriction was addressed and confirmed with the involvement of a supportive family member \
(contacted with patient consent), and de-escalation techniques were employed throughout this \
60-minute encounter. The patient was assessed as safe for continued outpatient care at the \
conclusion of the session — the least restrictive level of care appropriate given resolution of \
imminent risk — and was scheduled for a 24-hour follow-up contact to monitor continued \
stabilization. Given the patient's presentation of high distress under complex, life-threatening \
circumstances demanding immediate clinical attention, CPT code 90839 is medically necessary and \
clinically appropriate."""


def build_justification_fewshot_messages():
    return [
        {"role": "user", "content": FEWSHOT_STANDARD_INPUT},
        {"role": "assistant", "content": FEWSHOT_STANDARD_OUTPUT},
        {"role": "user", "content": FEWSHOT_CRISIS_INPUT},
        {"role": "assistant", "content": FEWSHOT_CRISIS_OUTPUT},
    ]


def format_similar_cases_context(similar_cases):
    if not similar_cases:
        return "No comparable historical cases were retrieved."
    lines = []
    for case in similar_cases[:3]:
        lines.append(
            f"- Diagnosis: {case.get('Primary_Diagnosis')}, Trajectory: {case.get('Trajectory_Type')}, "
            f"Progress at that point: {case.get('Progress')}, similarity: {case.get('similarity', 0):.2f}"
        )
    return "\n".join(lines)


def build_justification_prompt(session_result, transcript):
    diagnosis = session_result.get("Primary_Diagnosis", "Unknown")
    modality_hint = DIAGNOSIS_MODALITY_HINTS.get(diagnosis, "general evidence-based supportive therapy techniques")
    previous = session_result.get("Previous_Session_Summary")
    progress_text = (
        f"Progress vs. previous session: {session_result.get('Progress')} "
        f"(previous Risk_Score={previous['Risk_Score']}, previous Dynamic={previous['Dynamic']})"
        if previous else
        "This is the patient's first recorded session (intake) — no prior session to compare against; "
        "frame ongoing monitoring in terms of the plan for future sessions rather than a completed change."
    )
    cpt_guidance = build_cpt_guidance(session_result["Crisis_Flag"], session_result["Session_Duration_minutes"])

    return f"""Session data:
- Primary_Diagnosis: {diagnosis}
- Typical evidence-based treatment approach for this diagnosis (adapt to what's actually in the transcript, do not recite verbatim): {modality_hint}
- Risk_Score: {session_result['Predicted_Risk_Score']} ({session_result['Risk_Level']})
- Crisis_Flag: {session_result['Crisis_Flag']}
- Session_Duration: {session_result['Session_Duration_minutes']} minutes
- Assigned CPT Code: {session_result['Target_CPT_Code']}
- {progress_text}

{cpt_guidance}

Session transcript:
{transcript}

Write the Medical Necessity Justification paragraph for this session."""


def build_strategy_prompt(session_result):
    diagnosis = session_result.get("Primary_Diagnosis", "Unknown")
    modality_hint = DIAGNOSIS_MODALITY_HINTS.get(diagnosis, "general evidence-based supportive therapy techniques")
    previous = session_result.get("Previous_Session_Summary")
    previous_text = (
        f"Previous session: Risk_Score={previous['Risk_Score']}, Dynamic={previous['Dynamic']}"
        if previous else "This is the patient's first recorded session (intake) — no prior history."
    )

    return f"""Patient profile:
- Primary_Diagnosis: {diagnosis}
- Typical evidence-based focus areas for this diagnosis (adapt, do not recite verbatim): {modality_hint}

Current session metrics:
- Risk_Score: {session_result['Predicted_Risk_Score']} ({session_result['Risk_Level']})
- Crisis_Flag: {session_result['Crisis_Flag']}
- Dynamic (engagement style): {session_result['Predicted_Dynamic']}
- Progress (trajectory vs. previous session): {session_result['Progress']}

{previous_text}

Similar past cases (for general pattern context only):
{format_similar_cases_context(session_result.get('Similar_Cases'))}

Write the Next-Session Strategy paragraph — clinical recommendations for how the therapist \
should approach the upcoming session with this patient."""


# --- Resolve Primary_Diagnosis for sessions where recsys.py didn't have it --
def resolve_primary_diagnosis(patient_id, provided_diagnosis, previous_session, similar_cases):
    """Priority: explicit clinician input > patient's own history > RAG majority
    vote among retrieved similar cases (mpnet's Precision@5 was its strongest
    metric at 0.452 — a reasonable fallback, not blind guessing, but weaker
    than an explicitly known value)."""
    if provided_diagnosis:
        return provided_diagnosis, "clinician_provided"
    if previous_session:
        return previous_session["Primary_Diagnosis"], "patient_history"
    if similar_cases:
        diagnoses = [c["Primary_Diagnosis"] for c in similar_cases]
        majority = max(set(diagnoses), key=diagnoses.count)
        return majority, "inferred_from_similar_cases"
    return "Unknown", "unresolved"


# --- Main entry point: ties recsys.py's output to Part 4's generation ------
def generate_clinical_outputs(session_result, transcript):
    """Takes recsys.analyze_new_session()'s output dict + the raw transcript,
    generates both GenAI outputs, and returns them alongside the resolved
    diagnosis and its source for transparency."""
    diagnosis, diagnosis_source = resolve_primary_diagnosis(
        session_result["Patient_ID"],
        session_result.get("Primary_Diagnosis"),  # None unless the clinician overrode it in the UI
        session_result.get("Previous_Session_Summary"),
        session_result.get("Similar_Cases"),
    )
    session_result = {**session_result, "Primary_Diagnosis": diagnosis}

    justification = call_model_with_retry_sync(
        messages=[
            {"role": "system", "content": JUSTIFICATION_SYSTEM_PROMPT},
            *build_justification_fewshot_messages(),
            {"role": "user", "content": build_justification_prompt(session_result, transcript)},
        ],
        max_tokens=320, temperature=0.3,
    )

    strategy = call_model_with_retry_sync(
        messages=[
            {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
            {"role": "user", "content": build_strategy_prompt(session_result)},
        ],
        max_tokens=300, temperature=0.65,
    )

    return {
        "Medical_Necessity_Justification": justification,
        "Next_Session_Strategy": strategy,
        "Primary_Diagnosis_Used": diagnosis,
        "Primary_Diagnosis_Source": diagnosis_source,
    }
