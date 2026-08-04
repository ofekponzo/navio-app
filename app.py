"""
NavIO — app.py
Gradio Space entry point: an EMR-style clinician dashboard wrapping the
recsys.py (Part 3) and generation.py (Part 4) pipelines.

Architecture (see prior design discussion in the project):
  - A binary login gate implemented as two sibling gr.Group()s toggled via
    `visible=True/False` — deliberately NOT a nav item, since "Login" sitting
    as a peer option next to "Dashboard" would be the wrong metaphor for an
    all-or-nothing gate.
  - Post-login navigation is a HAND-BUILT sidebar: a dark gr.Column of plain
    gr.Button()s (Dashboard/Patients/Sessions) styled via elem_classes, next
    to a content gr.Column holding three gr.Group() "pages" toggled via
    visible=True/False — NOT gr.Tabs(). Gradio 6.x's Tabs component renders
    with Svelte-scoped, hashed internal CSS classes (verified directly by
    inspecting the compiled frontend bundle) that aren't reliably targetable
    from custom CSS, so restyling Tabs into a true left sidebar isn't robust.
    A hand-built sidebar uses only components/classes this file controls.
  - Every place a matplotlib figure or rich HTML card needs styling, the
    styling is bound via elem_classes/elem_id pointing at custom CSS in
    CUSTOM_CSS — never via raw HTML with embedded click handlers. Gradio
    cannot attach a Python callback to an arbitrary HTML element, so any
    control the clinician needs to click is a real Gradio component
    (gr.Button, gr.Dataframe with .select(), etc.); HTML is used only where
    the content is purely visual (cards, badges, the calendar day-strip).
  - There is no separate hidden "workspace" page — the Sessions page IS the
    analysis workspace (metrics/rings, both clinical graphs, justification,
    CPT badge, strategy), reached either directly or pre-filled via "Open"
    (a Dashboard appointment) / "Add New Session" (Patients profile). The
    exact same rendering also appears, read-only, inside the Patients page
    when a clinician clicks a past session — see render_session_dashboard()
    — so historical review never opens a popup/modal, only this page's own
    inline "Session Detail" section.
  - A single gr.State() dict carries all session-scoped data (logged-in
    doctor, doctor-scoped patient list, current sidebar view, and a
    session_log of newly-analyzed sessions/newly-registered patients).
    Critical for correctness: Gradio serves one process to every visitor, so
    anything stored as a plain global Python variable would leak across
    concurrent users. Nothing written during a session ever mutates
    recsys.df or retrieval_index — new data only ever lives in this
    per-session state dict.
  - Models, the dataset, and the retrieval index all load once at import
    time via `import recsys` / `import generation` — never inside a callback.

Known scope boundary: new patients/sessions logged during a browser session
live only in that session's gr.State and are lost on refresh. This is a
demo-appropriate choice, not an oversight — persisting back to the HF
dataset repo on every submission would be a much heavier addition than this
phase calls for.

All interface copy is English-only by project requirement.
"""

try:
    # Must be the very first import in the file, before anything that touches
    # CUDA (torch, sentence-transformers, transformers — all pulled in by
    # recsys.py below). HF's Docker template bundles this package into every
    # Gradio Space's environment; if it's imported after CUDA has already been
    # initialized, its background hot-reload watcher thread crashes with
    # "CUDA has been initialized before importing the `spaces` package."
    import spaces
    HAS_SPACES = True
except ImportError:
    spaces = None
    HAS_SPACES = False  # plain cpu-basic Space — nothing else to do

import gradio as gr
import pandas as pd
from datetime import datetime
from functools import partial

import auth
import recsys
import generation
import ui_helpers

DIAGNOSIS_CHOICES = ["Auto-detect"] + sorted(generation.DIAGNOSIS_MODALITY_HINTS.keys())
ALL_PATIENT_IDS = recsys.get_all_patient_ids()
MAX_SCHEDULE_SLOTS = 8


if HAS_SPACES:
    @spaces.GPU
    def _zerogpu_startup_probe():
        """NavIO's entire pipeline (mpnet embeddings, DistilBERT sentiment,
        all sklearn models) runs on CPU by design — nothing here needs a GPU.
        This function exists purely because HF's ZeroGPU hardware tier
        requires at least one @spaces.GPU-decorated function to be detected
        and invoked at startup, or the Space fails with 'No @spaces.GPU
        function detected during startup.' It is never called anywhere in
        the real request path."""
        return True


# ==============================================================================
# State helpers
# ==============================================================================
def default_state():
    return {
        "doctor": None,
        "assigned_patient_ids": [],
        "today_schedule": [],
        "session_log": {},          # {patient_id: [session_record, ...]} added this browser session
        "new_patients": [],         # patient_ids registered via Intake this session
        "selected_profile_patient": None,
        "current_profile_history": [],
        "pending_patient_id": None,
        "current_view": "dashboard",
    }


def get_full_history(patient_id, state):
    dataset_history = recsys.retrieval_index.get_patient_history(patient_id)
    session_history = state["session_log"].get(patient_id, [])
    return ui_helpers.merge_history(dataset_history, session_history)


def known_patient_choices(state):
    return sorted(set(state["assigned_patient_ids"]) | set(state["new_patients"]))


def _get_field(record, *keys, default=None):
    for key in keys:
        val = record.get(key)
        if val:
            return val
    return default


# ==============================================================================
# Sidebar navigation (hand-built — see module docstring for why)
# ==============================================================================
NAV_ACTIVE_CLASSES = ["navio-nav-btn", "navio-nav-btn-active"]
NAV_INACTIVE_CLASSES = ["navio-nav-btn"]


def _view_updates(view_name):
    return (
        gr.update(visible=view_name == "dashboard"),
        gr.update(visible=view_name == "patients"),
        gr.update(visible=view_name == "sessions"),
        gr.update(elem_classes=NAV_ACTIVE_CLASSES if view_name == "dashboard" else NAV_INACTIVE_CLASSES),
        gr.update(elem_classes=NAV_ACTIVE_CLASSES if view_name == "patients" else NAV_INACTIVE_CLASSES),
        gr.update(elem_classes=NAV_ACTIVE_CLASSES if view_name == "sessions" else NAV_INACTIVE_CLASSES),
    )


def switch_view(view_name, state):
    state = {**state, "current_view": view_name}
    return (*_view_updates(view_name), state)


def _schedule_slot_updates(schedule):
    """Returns 2 updates per fixed agenda-row slot (row visibility, card
    HTML) plus one for the 'no appointments' empty state — 2*MAX_SCHEDULE_
    SLOTS + 1 total, in the exact order the layout below wires them up."""
    updates = []
    for i in range(MAX_SCHEDULE_SLOTS):
        if i < len(schedule):
            updates.append(gr.update(visible=True))
            updates.append(gr.update(value=ui_helpers.format_agenda_row_html(schedule[i])))
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(value=""))
    updates.append(gr.update(visible=len(schedule) == 0))
    return updates


def _empty_schedule_updates():
    return [gr.update()] * (2 * MAX_SCHEDULE_SLOTS + 1)


# ==============================================================================
# Login
# ==============================================================================
def handle_login(name, id_number, state):
    doctor = auth.authenticate(name, id_number)
    if doctor is None:
        return (
            gr.update(visible=True),   # login_group
            gr.update(visible=False),  # main_group
            state,
            gr.update(value="**Login failed** — check the doctor name and ID number and try again.", visible=True),
            gr.update(), gr.update(), gr.update(),  # doctor_footer, roster_df, calendar_strip (unchanged)
            *_empty_schedule_updates(),
        )

    state = {**default_state(), "doctor": doctor}
    state["assigned_patient_ids"] = auth.get_assigned_patient_ids(doctor["doctor_id"], ALL_PATIENT_IDS)
    state["today_schedule"] = auth.generate_mock_schedule(doctor["doctor_id"], state["assigned_patient_ids"])

    roster_df = pd.DataFrame({"Patient_ID": known_patient_choices(state)})
    doctor_footer_html = (
        f"<div><strong>{doctor['name']}</strong></div>"
        f"<div>{len(state['assigned_patient_ids'])} patients on caseload</div>"
    )

    return (
        gr.update(visible=False),  # login_group
        gr.update(visible=True),   # main_group
        state,
        gr.update(value="", visible=False),  # login_error
        gr.update(value=doctor_footer_html),
        roster_df,
        gr.update(value=ui_helpers.build_calendar_strip_html(state["today_schedule"])),
        *_schedule_slot_updates(state["today_schedule"]),
    )


# ==============================================================================
# Dashboard page
# ==============================================================================
def start_session_from_slot(slot_index, state):
    schedule = state.get("today_schedule", [])
    if slot_index >= len(schedule):
        return (gr.update(), *switch_view(state.get("current_view", "dashboard"), state))
    patient_id = schedule[slot_index]["Patient_ID"]
    state = {**state, "pending_patient_id": patient_id}
    return (gr.update(value=patient_id), *switch_view("sessions", state))


# ==============================================================================
# Patients page (master-detail, with a full inline "Session Detail" dashboard
# — no popups: clicking a past session renders the same rich view Sessions
# uses, right on this page, via render_session_dashboard()).
# ==============================================================================
def filter_roster(search_text, state):
    choices = known_patient_choices(state)
    if search_text:
        choices = [c for c in choices if search_text.strip().lower() in c.lower()]
    return pd.DataFrame({"Patient_ID": choices})


def _reset_detail_view():
    """Collapses the inline Session Detail block back to its empty state —
    used whenever the selected patient changes, since a session detail from
    the PREVIOUS patient should never linger on screen."""
    return gr.update(visible=True), gr.update(visible=False)


def render_patient_profile(patient_id, state):
    if not patient_id:
        empty_msg, detail_group = _reset_detail_view()
        return (
            gr.update(value="Search or select a patient from the roster on the left."),
            pd.DataFrame(columns=["Session #", "Diagnosis", "Risk Score", "Risk Level", "Dynamic", "Progress", "CPT Code", "Crisis"]),
            ui_helpers.build_risk_trajectory_figure([], None),
            empty_msg, detail_group,
            state,
        )

    history = get_full_history(patient_id, state)
    state = {**state, "selected_profile_patient": patient_id, "current_profile_history": history}

    if not history:
        header = f"### {patient_id}\n\n" + ui_helpers.no_history_html()
    else:
        header = f"### {patient_id}  &middot;  {len(history)} session(s) on file"

    history_df = ui_helpers.format_patient_history_df(history)
    trajectory_fig = ui_helpers.build_risk_trajectory_figure(history, None)
    empty_msg, detail_group = _reset_detail_view()
    return gr.update(value=header), history_df, trajectory_fig, empty_msg, detail_group, state


def on_roster_select(evt: gr.SelectData, state):
    roster = known_patient_choices(state)
    row_index = evt.index[0]
    patient_id = roster[row_index] if row_index < len(roster) else None
    return render_patient_profile(patient_id, state)


def render_session_dashboard(record):
    """Builds the full read-only 'Session Detail' dashboard for ONE
    historical record: metric/ring cards, Graph 1 (with that session
    highlighted), Graph 2 (parsed from the stored transcript, if any), CPT
    badge, justification, and strategy — the same visual language as the
    live Sessions workspace, reused here instead of a popup/modal."""
    history = record.get("_patient_history", [])
    cards_html = ui_helpers.format_metric_cards_html(record)
    graph1 = ui_helpers.build_risk_trajectory_figure(history, None, highlight_session_number=record.get("Session_Number"))

    duration = _get_field(record, "Session_Duration_minutes", "Session_Duration", default=45)
    transcript = record.get("Transcript") or ""
    timeline_turns = recsys.parse_intrasession_timeline(transcript)
    graph2 = ui_helpers.build_intrasession_timeline_figure(timeline_turns, duration, bool(record.get("Crisis_Flag")))

    cpt_badge_html = ui_helpers.format_cpt_badge_html(record.get("Target_CPT_Code", "—"), bool(record.get("Crisis_Flag")))
    justification = record.get("Medical_Necessity_Justification") or "No stored justification for this historical session."
    strategy_html = ui_helpers.format_strategy_html(record.get("Next_Session_Strategy") or "No stored strategy for this historical session.")

    return cards_html, graph1, graph2, justification, cpt_badge_html, strategy_html


def on_profile_history_select(evt: gr.SelectData, state):
    records = state.get("current_profile_history", [])
    row_index = evt.index[0]
    if row_index >= len(records):
        return (gr.update(visible=True), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    record = {**records[row_index], "_patient_history": records}
    cards_html, graph1, graph2, justification, cpt_badge_html, strategy_html = render_session_dashboard(record)
    return (
        gr.update(visible=False),  # empty-state message
        gr.update(visible=True),   # detail group
        cards_html, graph1, graph2, justification, cpt_badge_html, strategy_html,
    )


def register_new_patient(new_patient_id, state):
    new_patient_id = (new_patient_id or "").strip()
    if not new_patient_id:
        error = gr.update(value="Enter a Patient ID before registering.", visible=True)
        return gr.update(), state, error, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    if new_patient_id in ALL_PATIENT_IDS or new_patient_id in state["new_patients"]:
        error = gr.update(value=f"Patient ID '{new_patient_id}' already exists.", visible=True)
        return gr.update(), state, error, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    state = {**state, "new_patients": state["new_patients"] + [new_patient_id]}
    roster_df = pd.DataFrame({"Patient_ID": known_patient_choices(state)})
    header, history_df, fig, empty_msg, detail_group, state = render_patient_profile(new_patient_id, state)
    no_error = gr.update(value="", visible=False)
    return roster_df, state, no_error, header, history_df, fig, empty_msg, detail_group


def add_session_from_profile(state):
    patient_id = state.get("selected_profile_patient")
    if not patient_id:
        return (gr.update(), *switch_view(state.get("current_view", "patients"), state))
    state = {**state, "pending_patient_id": patient_id}
    return (gr.update(value=patient_id), *switch_view("sessions", state))


# ==============================================================================
# Sessions page (the analysis workspace — shared by both entry points AND
# direct/standalone use)
# ==============================================================================
def _run_pipeline(transcript, patient_id, duration_minutes, diagnosis_choice):
    patient_id = (patient_id or "").strip()
    if not patient_id:
        raise gr.Error("Enter or select a Patient ID before analyzing a session.")
    if not transcript or not transcript.strip():
        raise gr.Error("Paste or type the session transcript before analyzing.")

    is_known = patient_id in ALL_PATIENT_IDS
    session_result = recsys.analyze_new_session(transcript, patient_id, duration_minutes)

    diagnosis_override = None if diagnosis_choice in (None, "Auto-detect") else diagnosis_choice
    session_result["Primary_Diagnosis"] = diagnosis_override

    try:
        generation_result = generation.generate_clinical_outputs(session_result, transcript)
    except generation.QuotaExceededError:
        raise gr.Error("The generation model's usage quota was exceeded. Please try again later.")
    except Exception as e:
        raise gr.Error(f"Generation failed: {e}")

    session_result["Primary_Diagnosis"] = generation_result["Primary_Diagnosis_Used"]

    status_note = ""
    if not is_known and session_result["session_context"] == "intake":
        status_note = f"*New patient — '{patient_id}' was not found and has been auto-registered as an intake session.*"

    cards_html = ui_helpers.format_metric_cards_html(session_result)

    # Graph 1 — cross-session trajectory (prior recorded sessions + this one).
    history_for_plot = recsys.retrieval_index.get_patient_history(patient_id)
    trajectory_fig = ui_helpers.build_risk_trajectory_figure(history_for_plot, session_result)

    # Graph 2 — intra-session timeline, parsed live from THIS transcript's
    # [MM:SS] markers, rescaled across the full session duration. Presentation
    # -only; does not feed the trained models.
    timeline_turns = recsys.parse_intrasession_timeline(transcript)
    intrasession_fig = ui_helpers.build_intrasession_timeline_figure(
        timeline_turns, session_result["Session_Duration_minutes"], session_result["Crisis_Flag"],
    )

    cpt_badge_html = ui_helpers.format_cpt_badge_html(session_result["Target_CPT_Code"], session_result["Crisis_Flag"])
    strategy_html = ui_helpers.format_strategy_html(generation_result["Next_Session_Strategy"])

    return (
        status_note, cards_html, trajectory_fig, intrasession_fig,
        generation_result["Medical_Necessity_Justification"], cpt_badge_html, strategy_html,
        session_result, patient_id, is_known, transcript, generation_result["Next_Session_Strategy"],
    )


def run_session_analysis(transcript, patient_id, duration_minutes, diagnosis_choice, state):
    """Used by the 'Analyze Session' button — persists the result into this
    browser session's in-memory log (including the transcript itself and the
    generated strategy text) so that later reopening this session from the
    Patients page's inline Session Detail can render its full dashboard,
    Graph 2 included, with no popup."""
    (status_note, cards_html, trajectory_fig, intrasession_fig, justification,
     cpt_badge_html, strategy_html, session_result, patient_id, is_known,
     transcript_text, strategy_text) = _run_pipeline(transcript, patient_id, duration_minutes, diagnosis_choice)

    new_state = {
        **state,
        "session_log": {**state["session_log"]},
        "new_patients": list(state["new_patients"]),
    }
    if not is_known and patient_id not in new_state["new_patients"]:
        new_state["new_patients"] = new_state["new_patients"] + [patient_id]

    logged_record = {
        "Session_Number": ui_helpers.next_session_number(get_full_history(patient_id, state)),
        "Primary_Diagnosis": session_result["Primary_Diagnosis"],
        "Risk_Score": session_result["Predicted_Risk_Score"],
        "Risk_Level": session_result["Risk_Level"],
        "Dynamic": session_result["Predicted_Dynamic"],
        "Progress": session_result["Progress"],
        "Target_CPT_Code": session_result["Target_CPT_Code"],
        "Crisis_Flag": session_result["Crisis_Flag"],
        "Crisis_Probability": session_result["Crisis_Probability"],
        "Medical_Necessity_Justification": justification,
        "Next_Session_Strategy": strategy_text,
        "Transcript": transcript_text,
        "Session_Duration_minutes": session_result["Session_Duration_minutes"],
    }
    new_state["session_log"][patient_id] = new_state["session_log"].get(patient_id, []) + [logged_record]

    return status_note, cards_html, trajectory_fig, intrasession_fig, justification, cpt_badge_html, strategy_html, new_state


def quick_start_analyze(patient_id, transcript, duration_minutes, diagnosis_choice):
    """Used by the Quick Starters — runs the real pipeline live so evaluators
    see it actually work, but doesn't touch session state (a demo run isn't
    meant to permanently add to a clinician's caseload)."""
    status_note, cards_html, trajectory_fig, intrasession_fig, justification, cpt_badge_html, strategy_html, *_ = \
        _run_pipeline(transcript, patient_id, duration_minutes, diagnosis_choice)
    return status_note, cards_html, trajectory_fig, intrasession_fig, justification, cpt_badge_html, strategy_html


def build_quick_starters():
    """Built dynamically from real, currently-loaded patient IDs rather than
    hardcoded guesses — guarantees the 'existing patient' examples actually
    have retrievable history in the live dataset."""
    existing_id_1 = ALL_PATIENT_IDS[0]
    existing_id_2 = ALL_PATIENT_IDS[1] if len(ALL_PATIENT_IDS) > 1 else ALL_PATIENT_IDS[0]
    new_id = "P_NEW_DEMO_001"

    standard_transcript = (
        "[00:00] Dr. Carter: How has your week been?\n"
        "[00:08] Patient: A bit better, honestly. I managed to go to the grocery store without leaving early this time.\n"
        "[00:20] Dr. Carter: That's real progress. What felt different about it?\n"
        "[00:30] Patient: I used the breathing exercise we practiced before I felt overwhelmed.\n"
        "[00:45] Dr. Carter: Let's build on that — this week, try applying it in one more situation you've been avoiding."
    )
    crisis_transcript = (
        "[00:00] Dr. Carter: I'm glad you reached out. Can you tell me what's happening right now?\n"
        "[00:06] Patient: I don't feel safe. I've been thinking about hurting myself and I have a plan.\n"
        "[00:18] Dr. Carter: Thank you for telling me — that took courage. Let's work through a safety plan together right now.\n"
        "[00:35] Patient: Okay. I'm scared but I don't want to be alone with this.\n"
        "[00:50] Dr. Carter: You won't be. Let's also talk about who from your support network we can bring in tonight."
    )
    intake_transcript = (
        "[00:00] Dr. Carter: Welcome — this is our first session, so I'd like to understand what's been going on.\n"
        "[00:12] Patient: I've had trouble sleeping for months and I feel on edge most of the day.\n"
        "[00:28] Dr. Carter: How has that been affecting your day-to-day life?\n"
        "[00:36] Patient: I've been avoiding calls from friends and I'm behind on things at work.\n"
        "[00:50] Dr. Carter: That's helpful context. Let's start mapping out what's triggering that on-edge feeling."
    )

    return [
        [existing_id_1, standard_transcript, 45, "Auto-detect"],
        [existing_id_2, crisis_transcript, 60, "Auto-detect"],
        [new_id, intake_transcript, 45, "Auto-detect"],
    ]


# ==============================================================================
# Design system CSS — palette/type matched to the reference enterprise-EMR
# mockup (dark navy sidebar, teal primary accent, blue secondary accent,
# Hanken Grotesk body / IBM Plex Mono for uppercase labels and data figures).
# All rules below target either plain HTML tags (stable across Gradio
# versions) or elem_id/elem_classes THIS FILE assigns — never guessed-at
# Gradio-internal class names (see the Tabs note in the module docstring).
# ==============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root {
    --navio-bg: #f5f7fb;
    --navio-card: #ffffff;
    --navio-border: #e7edf4;
    --navio-text: #14223b;
    --navio-text-secondary: #6b7b90;
    --navio-text-muted: #9aa7b8;
    --navio-accent: #06b4ab;
    --navio-accent-hover: #0a857e;
    --navio-accent-soft: #e6f7f5;
    --navio-secondary: #3b6fe0;
    --navio-sidebar-bg: #0f1b30;
    --navio-sidebar-border: #1c2c49;
    --font-body: 'Hanken Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

.gradio-container {
    background: var(--navio-bg) !important;
    font-family: var(--font-body) !important;
}
.gradio-container * { font-family: var(--font-body); }

/* ---- Clean separation between sidebar and main content ---- */
.navio-app-row { gap: 24px !important; align-items: flex-start !important; }
#navio_main_content { padding: 4px 2px 40px !important; }

/* ---- Hand-built sidebar ---- */
#navio_sidebar {
    background: var(--navio-sidebar-bg) !important;
    border-radius: 18px !important;
    padding: 22px 14px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 3px !important;
    min-height: 76vh;
    box-shadow: 0 10px 30px rgba(9,18,38,0.25);
}
.navio-brand-box {
    color: #fff; font-weight: 800; font-size: 20px; letter-spacing: 0.01em;
    padding: 6px 12px 26px;
}
.navio-brand-box span { color: var(--navio-accent); }

.navio-nav-btn button {
    background: transparent !important;
    color: #9aa7c4 !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    box-shadow: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    padding: 11px 14px !important;
    font-weight: 600 !important;
    font-size: 14.5px !important;
    transition: all 0.15s ease !important;
}
.navio-nav-btn button:hover { background: rgba(255,255,255,0.06) !important; color: #fff !important; }
.navio-nav-btn-active button {
    background: rgba(6,180,171,0.14) !important;
    color: #fff !important;
    border-left: 3px solid var(--navio-accent) !important;
}

.navio-sidebar-footer {
    margin-top: auto !important; padding: 16px 12px 4px !important;
    border-top: 1px solid var(--navio-sidebar-border) !important;
    color: #9aa7c4 !important; font-size: 12.5px !important; line-height: 1.6;
}
.navio-sidebar-footer strong { color: #fff; font-size: 13.5px; }

/* ---- Page headers ---- */
.navio-page-title { font-size: 22px; font-weight: 800; color: var(--navio-text); letter-spacing: -0.01em; margin: 4px 0 2px; }
.navio-page-subtitle { font-size: 13px; color: var(--navio-text-secondary); margin-bottom: 20px; line-height: 1.55; max-width: 720px; }
.navio-section-label {
    font-family: var(--font-mono); font-size: 11px; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--navio-text-secondary); font-weight: 600; margin: 22px 0 10px;
}

/* ---- Generic elevated card wrapper (applied via elem_classes) ---- */
.navio-card {
    background: var(--navio-card) !important;
    border: 1px solid var(--navio-border) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 6px 20px rgba(16,34,59,0.05) !important;
}

/* ---- Calendar day-strip + agenda list ---- */
.navio-calendar-scroll { max-height: 420px; overflow-y: auto; }
.navio-agenda-row { align-items: center !important; }
.navio-agenda-open-btn button {
    background: var(--navio-accent) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
    font-size: 13px !important;
}
.navio-agenda-open-btn button:hover { background: var(--navio-accent-hover) !important; }

/* ---- Primary action buttons throughout ---- */
button.primary, .gradio-container button[class*="primary"] {
    background: var(--navio-accent) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
button.primary:hover, .gradio-container button[class*="primary"]:hover {
    background: var(--navio-accent-hover) !important;
}

/* ---- Tables: plain HTML <table>/<th>/<td> tags are stable across Gradio
   versions even when component wrapper classes aren't, so this targets the
   semantic elements directly rather than guessing Gradio's internal classes. ---- */
.gradio-container table { border-collapse: collapse !important; font-size: 13px !important; }
.gradio-container table thead th {
    background: #fafbfd !important; color: var(--navio-text-secondary) !important;
    font-weight: 700 !important; text-transform: uppercase; font-size: 10.5px !important;
    letter-spacing: 0.05em; font-family: var(--font-mono) !important;
}
.gradio-container table td, .gradio-container table th {
    padding: 10px 14px !important; border-bottom: 1px solid var(--navio-border) !important;
}
"""

# ==============================================================================
# Layout
# ==============================================================================
with gr.Blocks(title="NavIO") as demo:
    app_state = gr.State(value=default_state())

    # --- Login screen ---------------------------------------------------
    with gr.Group(visible=True) as login_group:
        gr.Markdown("## NavIO\n#### Clinical Intelligence Platform — Clinician Login")
        gr.Markdown(
            "*Demo login for illustration only — not production authentication. "
            "Use one of the demo clinician accounts (see project README).*"
        )
        login_name = gr.Textbox(label="Doctor Name", placeholder="e.g. Dr. Sarah Cohen")
        login_id = gr.Textbox(label="ID Number", placeholder="Demo ID number")
        login_button = gr.Button("Log In", variant="primary")
        login_error = gr.Markdown(value="", visible=False)

    # --- Main app ---------------------------------------------------------
    with gr.Group(visible=False) as main_group:
        with gr.Row(elem_classes=["navio-app-row"]):

            # ---------------- Sidebar ----------------
            with gr.Column(elem_id="navio_sidebar", scale=0, min_width=220):
                gr.HTML('<div class="navio-brand-box">NAV<span>IO</span></div>')
                nav_dashboard_btn = gr.Button("Dashboard", elem_classes=NAV_ACTIVE_CLASSES)
                nav_patients_btn = gr.Button("Patients", elem_classes=NAV_INACTIVE_CLASSES)
                nav_sessions_btn = gr.Button("Sessions", elem_classes=NAV_INACTIVE_CLASSES)
                doctor_footer = gr.HTML(value="", elem_classes=["navio-sidebar-footer"])

            # ---------------- Main content ----------------
            with gr.Column(elem_id="navio_main_content", scale=1):

                # ---- Dashboard page ----
                with gr.Group(visible=True) as dashboard_page:
                    gr.HTML('<div class="navio-page-title">Today\'s Schedule</div>')
                    gr.HTML(
                        '<div class="navio-page-subtitle">NavIO\'s dataset models historical therapy '
                        'sessions, not future bookings — this schedule is a demo simulation for '
                        'illustration, distinct from the real session archive in each patient\'s profile.</div>'
                    )
                    gr.HTML('<div class="navio-section-label">Day Overview</div>')
                    with gr.Group(elem_classes=["navio-card", "navio-calendar-scroll"]):
                        calendar_strip = gr.HTML(value=ui_helpers.build_calendar_strip_html([]))
                    gr.HTML('<div class="navio-section-label">Today\'s Appointments</div>')
                    schedule_slots = []
                    for _ in range(MAX_SCHEDULE_SLOTS):
                        with gr.Row(visible=False, elem_classes=["navio-agenda-row"]) as slot_row:
                            with gr.Column(scale=5):
                                slot_html = gr.HTML()
                            with gr.Column(scale=1, min_width=110):
                                slot_button = gr.Button("Open →", elem_classes=["navio-agenda-open-btn"])
                        schedule_slots.append((slot_row, slot_html, slot_button))
                    no_schedule_msg = gr.HTML(value=ui_helpers.no_schedule_html(), visible=False)

                # ---- Patients page ----
                with gr.Group(visible=False) as patients_page:
                    gr.HTML('<div class="navio-page-title">Patient Directory</div>')
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["navio-card"]):
                            search_box = gr.Textbox(label="Search by Patient ID", placeholder="e.g. P0001")
                            roster_df = gr.Dataframe(headers=["Patient_ID"], interactive=False)
                            with gr.Accordion("Register New Patient (Intake)", open=False):
                                new_patient_id_box = gr.Textbox(label="New Patient ID", placeholder="e.g. P_NEW_001")
                                register_button = gr.Button("Register")
                                register_error = gr.Markdown(value="", visible=False)
                        with gr.Column(scale=2, elem_classes=["navio-card"]):
                            profile_header = gr.Markdown("Search or select a patient from the roster on the left.")
                            profile_history_df = gr.Dataframe(
                                headers=["Session #", "Diagnosis", "Risk Score", "Risk Level", "Dynamic", "Progress", "CPT Code", "Crisis"],
                                interactive=False,
                            )
                            profile_trajectory_plot = gr.Plot(label="Graph 1 — Cross-Session Progress & Trajectory")
                            add_session_profile_btn = gr.Button("Add New Session", variant="primary")

                    # ---- Session Detail — the FULL dashboard for a clicked
                    # past session, inline on this page. No popup/modal:
                    # clicking a row in profile_history_df above swaps the
                    # empty-state message for this block in place. ----
                    gr.HTML('<div class="navio-section-label">Session Detail</div>')
                    profile_detail_empty = gr.HTML(value=ui_helpers.no_session_selected_html())
                    with gr.Group(visible=False) as profile_detail_group:
                        with gr.Group(elem_classes=["navio-card"]):
                            profile_detail_cards = gr.HTML()
                        with gr.Row():
                            with gr.Column(scale=1, elem_classes=["navio-card"]):
                                profile_detail_graph1 = gr.Plot(label="Graph 1 — Cross-Session Progress & Trajectory")
                            with gr.Column(scale=1, elem_classes=["navio-card"]):
                                profile_detail_graph2 = gr.Plot(label="Graph 2 — Intra-Session Clinical Timeline")
                        gr.HTML('<div class="navio-section-label">Document &amp; Billing</div>')
                        with gr.Row():
                            with gr.Column(scale=2, elem_classes=["navio-card"]):
                                profile_detail_justification = gr.Textbox(label="Medical Necessity Justification", lines=6, interactive=False)
                            with gr.Column(scale=1, elem_classes=["navio-card"]):
                                profile_detail_cpt_badge = gr.HTML()
                        profile_detail_strategy = gr.HTML()

                # ---- Sessions page (the analysis workspace) ----
                with gr.Group(visible=False) as sessions_page:
                    gr.HTML('<div class="navio-page-title">New Session — Analysis &amp; Documentation</div>')
                    gr.HTML(
                        '<div class="navio-page-subtitle">Arriving from Dashboard or a Patient Profile '
                        'pre-fills the Patient ID below (still editable). Visiting this page directly '
                        'starts blank — typing an unrecognized Patient ID will auto-register it as a '
                        'new intake patient on submit.</div>'
                    )

                    # ---- Top: input controls ----
                    with gr.Row(elem_classes=["navio-card"]):
                        session_patient_id = gr.Dropdown(
                            label="Patient ID", choices=ALL_PATIENT_IDS, allow_custom_value=True,
                        )
                        session_duration = gr.Number(label="Session Duration (minutes)", value=45, minimum=1)
                        session_diagnosis = gr.Dropdown(label="Primary Diagnosis (override)", choices=DIAGNOSIS_CHOICES, value="Auto-detect")
                    session_transcript = gr.Textbox(
                        label="Session Transcript", lines=8,
                        placeholder="Paste or type the session transcript here (use [MM:SS] markers, e.g. \"[00:12] Dr. Carter: ...\", so Graph 2 can plot the intra-session timeline)...",
                    )
                    analyze_button = gr.Button("Analyze Session", variant="primary")

                    # Output components are created with render=False so they can be
                    # referenced by gr.Examples' outputs= below (Gradio needs the actual
                    # object, not a forward reference), while still visually placing the
                    # Quick Starters table above the results via explicit .render() calls.
                    session_status = gr.Markdown("", render=False)
                    session_cards = gr.HTML(render=False)
                    session_trajectory_plot = gr.Plot(label="Graph 1 — Cross-Session Progress & Trajectory", render=False)
                    session_intrasession_plot = gr.Plot(label="Graph 2 — Intra-Session Clinical Timeline", render=False)
                    justification_box = gr.Textbox(label="Medical Necessity Justification", lines=8, interactive=False, render=False)
                    cpt_badge = gr.HTML(render=False)
                    strategy_output = gr.HTML(label="Next-Session Strategy", render=False)

                    # fn+outputs here means clicking an example doesn't just pre-fill the
                    # inputs — it runs the live pipeline immediately, satisfying "1-click
                    # and see an example of how the application works."
                    gr.Examples(
                        examples=build_quick_starters(),
                        inputs=[session_patient_id, session_transcript, session_duration, session_diagnosis],
                        outputs=[session_status, session_cards, session_trajectory_plot, session_intrasession_plot, justification_box, cpt_badge, strategy_output],
                        fn=quick_start_analyze,
                        cache_examples=False,
                        label="Quick Starters",
                    )

                    session_status.render()

                    # ---- Middle: hard metrics row + the two graphs ----
                    with gr.Group(elem_classes=["navio-card"]):
                        session_cards.render()
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["navio-card"]):
                            session_trajectory_plot.render()
                        with gr.Column(scale=1, elem_classes=["navio-card"]):
                            session_intrasession_plot.render()

                    # ---- Document & Billing section ----
                    gr.HTML('<div class="navio-section-label">Document &amp; Billing</div>')
                    with gr.Row():
                        with gr.Column(scale=2, elem_classes=["navio-card"]):
                            justification_box.render()
                        with gr.Column(scale=1, elem_classes=["navio-card"]):
                            cpt_badge.render()

                    # ---- Bottom: Next-Session Strategy (prominent, own accent card) ----
                    strategy_output.render()

    # ==========================================================================
    # Wiring
    # ==========================================================================
    schedule_slot_outputs = [comp for row, html, btn in schedule_slots for comp in (row, html)] + [no_schedule_msg]

    login_button.click(
        fn=handle_login,
        inputs=[login_name, login_id, app_state],
        outputs=[login_group, main_group, app_state, login_error, doctor_footer, roster_df, calendar_strip, *schedule_slot_outputs],
    )

    view_switch_outputs = [dashboard_page, patients_page, sessions_page, nav_dashboard_btn, nav_patients_btn, nav_sessions_btn, app_state]
    nav_dashboard_btn.click(fn=partial(switch_view, "dashboard"), inputs=[app_state], outputs=view_switch_outputs)
    nav_patients_btn.click(fn=partial(switch_view, "patients"), inputs=[app_state], outputs=view_switch_outputs)
    nav_sessions_btn.click(fn=partial(switch_view, "sessions"), inputs=[app_state], outputs=view_switch_outputs)

    for i, (slot_row, slot_html, slot_button) in enumerate(schedule_slots):
        slot_button.click(
            fn=partial(start_session_from_slot, i),
            inputs=[app_state],
            outputs=[session_patient_id, *view_switch_outputs],
        )

    search_box.change(fn=filter_roster, inputs=[search_box, app_state], outputs=[roster_df])
    roster_df.select(
        fn=on_roster_select,
        inputs=[app_state],
        outputs=[profile_header, profile_history_df, profile_trajectory_plot, profile_detail_empty, profile_detail_group, app_state],
    )
    profile_history_df.select(
        fn=on_profile_history_select,
        inputs=[app_state],
        outputs=[profile_detail_empty, profile_detail_group, profile_detail_cards, profile_detail_graph1,
                 profile_detail_graph2, profile_detail_justification, profile_detail_cpt_badge, profile_detail_strategy],
    )
    register_button.click(
        fn=register_new_patient,
        inputs=[new_patient_id_box, app_state],
        outputs=[roster_df, app_state, register_error, profile_header, profile_history_df, profile_trajectory_plot,
                 profile_detail_empty, profile_detail_group],
    )
    add_session_profile_btn.click(
        fn=add_session_from_profile,
        inputs=[app_state],
        outputs=[session_patient_id, *view_switch_outputs],
    )

    analyze_button.click(
        fn=run_session_analysis,
        inputs=[session_transcript, session_patient_id, session_duration, session_diagnosis, app_state],
        outputs=[session_status, session_cards, session_trajectory_plot, session_intrasession_plot, justification_box, cpt_badge, strategy_output, app_state],
    )


if __name__ == "__main__":
    if HAS_SPACES:
        _zerogpu_startup_probe()  # registers the dummy GPU function so ZeroGPU's startup check passes
    demo.launch(css=CUSTOM_CSS)
