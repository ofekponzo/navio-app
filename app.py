"""
NavIO — app.py
Gradio Space entry point: an EMR-style clinician dashboard wrapping the
recsys.py (Part 3) and generation.py (Part 4) pipelines.

Architecture (see prior design discussion in the project):
  - A binary login gate implemented as two sibling gr.Group()s toggled via
    `visible=True/False` — deliberately NOT gr.Tabs(), since "Login" sitting
    as a peer option next to "Dashboard" would be the wrong metaphor for an
    all-or-nothing gate.
  - Post-login navigation is a native gr.Tabs() (CSS-styled into a vertical
    strip to read as a sidebar) with three real tabs: Dashboard, Patients,
    Sessions. There is no separate hidden "workspace" tab — the Sessions tab
    IS the analysis workspace (metrics, crisis graph, justification, CPT
    badge, strategy), reached either directly or pre-filled via "Start
    Session" (Dashboard) / "Add New Session" (Patients profile pane).
  - A single gr.State() dict carries all session-scoped data (logged-in
    doctor, doctor-scoped patient list, and a session_log of newly-analyzed
    sessions / newly-registered patients). This is critical for correctness:
    Gradio serves one process to every visitor, so anything stored as a
    plain global Python variable would leak across concurrent users. Nothing
    written during a session ever mutates recsys.df or retrieval_index — new
    data only ever lives in this per-session state dict.
  - Models, the dataset, and the retrieval index all load once at import
    time via `import recsys` / `import generation` — never inside a callback.

Known scope boundary: new patients/sessions logged during a browser session
live only in that session's gr.State and are lost on refresh. This is a
demo-appropriate choice, not an oversight — persisting back to the HF
dataset repo on every submission would be a much heavier addition than this
phase calls for.
"""

import gradio as gr
import pandas as pd
from datetime import datetime

import auth
import recsys
import generation
import ui_helpers

DIAGNOSIS_CHOICES = ["Auto-detect"] + sorted(generation.DIAGNOSIS_MODALITY_HINTS.keys())
ALL_PATIENT_IDS = recsys.get_all_patient_ids()


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
    }


def get_full_history(patient_id, state):
    dataset_history = recsys.retrieval_index.get_patient_history(patient_id)
    session_history = state["session_log"].get(patient_id, [])
    return ui_helpers.merge_history(dataset_history, session_history)


def known_patient_choices(state):
    return sorted(set(state["assigned_patient_ids"]) | set(state["new_patients"]))


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
            gr.update(), gr.update(), gr.update(),  # doctor_label, schedule_df, roster_df (unchanged)
        )

    state = {**default_state(), "doctor": doctor}
    state["assigned_patient_ids"] = auth.get_assigned_patient_ids(doctor["doctor_id"], ALL_PATIENT_IDS)
    state["today_schedule"] = auth.generate_mock_schedule(doctor["doctor_id"], state["assigned_patient_ids"])

    schedule_df = pd.DataFrame(state["today_schedule"]) if state["today_schedule"] else pd.DataFrame(columns=["Time", "Patient_ID", "Status"])
    roster_df = pd.DataFrame({"Patient_ID": known_patient_choices(state)})

    return (
        gr.update(visible=False),  # login_group
        gr.update(visible=True),   # main_group
        state,
        gr.update(value="", visible=False),  # login_error
        gr.update(value=f"Logged in as **{doctor['name']}** &middot; {len(state['assigned_patient_ids'])} patients on caseload"),
        schedule_df,
        roster_df,
    )


# ==============================================================================
# Dashboard tab
# ==============================================================================
def on_dashboard_select(evt: gr.SelectData, state):
    row_index = evt.index[0]
    if row_index < len(state["today_schedule"]):
        patient_id = state["today_schedule"][row_index]["Patient_ID"]
        return patient_id, f"Selected: **{patient_id}**"
    return None, ""


def start_session_from_dashboard(selected_patient_id, state):
    if not selected_patient_id:
        return gr.update(), state, gr.update(selected="sessions")
    state = {**state, "pending_patient_id": selected_patient_id}
    return gr.update(value=selected_patient_id), state, gr.update(selected="sessions")


# ==============================================================================
# Patients tab (master-detail)
# ==============================================================================
def filter_roster(search_text, state):
    choices = known_patient_choices(state)
    if search_text:
        choices = [c for c in choices if search_text.strip().lower() in c.lower()]
    return pd.DataFrame({"Patient_ID": choices})


def render_patient_profile(patient_id, state):
    if not patient_id:
        return (
            gr.update(value="Search or select a patient from the roster on the left."),
            pd.DataFrame(columns=["Session #", "Diagnosis", "Risk Score", "Risk Level", "Dynamic", "Progress", "CPT Code", "Crisis"]),
            ui_helpers.build_risk_trajectory_figure([], None),
            gr.update(value="", visible=False),
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
    return gr.update(value=header), history_df, trajectory_fig, gr.update(value="", visible=False), state


def on_roster_select(evt: gr.SelectData, state):
    roster = known_patient_choices(state)
    row_index = evt.index[0]
    patient_id = roster[row_index] if row_index < len(roster) else None
    return render_patient_profile(patient_id, state)


def on_profile_history_select(evt: gr.SelectData, state):
    records = state.get("current_profile_history", [])
    row_index = evt.index[0]
    if row_index < len(records):
        return gr.update(value=ui_helpers.format_session_detail_html(records[row_index]), visible=True)
    return gr.update(value="", visible=False)


def register_new_patient(new_patient_id, state):
    new_patient_id = (new_patient_id or "").strip()
    if not new_patient_id:
        error = gr.update(value="Enter a Patient ID before registering.", visible=True)
        return gr.update(), state, error, gr.update(), gr.update(), gr.update(), gr.update()
    if new_patient_id in ALL_PATIENT_IDS or new_patient_id in state["new_patients"]:
        error = gr.update(value=f"Patient ID '{new_patient_id}' already exists.", visible=True)
        return gr.update(), state, error, gr.update(), gr.update(), gr.update(), gr.update()

    state = {**state, "new_patients": state["new_patients"] + [new_patient_id]}
    roster_df = pd.DataFrame({"Patient_ID": known_patient_choices(state)})
    header, history_df, fig, detail, state = render_patient_profile(new_patient_id, state)
    no_error = gr.update(value="", visible=False)
    return roster_df, state, no_error, header, history_df, fig, detail


def add_session_from_profile(state):
    patient_id = state.get("selected_profile_patient")
    if not patient_id:
        return gr.update(), state, gr.update()
    state = {**state, "pending_patient_id": patient_id}
    return gr.update(value=patient_id), state, gr.update(selected="sessions")


# ==============================================================================
# Sessions tab (the analysis workspace — shared by both entry points AND
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
    history_for_plot = recsys.retrieval_index.get_patient_history(patient_id)
    trajectory_fig = ui_helpers.build_risk_trajectory_figure(history_for_plot, session_result)
    cpt_badge_html = ui_helpers.format_cpt_badge_html(session_result["Target_CPT_Code"], session_result["Crisis_Flag"])
    strategy_html = ui_helpers.format_strategy_html(generation_result["Next_Session_Strategy"])

    return status_note, cards_html, trajectory_fig, generation_result["Medical_Necessity_Justification"], cpt_badge_html, strategy_html, session_result, patient_id, is_known


def run_session_analysis(transcript, patient_id, duration_minutes, diagnosis_choice, state):
    """Used by the 'Analyze Session' button — persists the result into this
    browser session's in-memory log so it shows up in Patients/Dashboard."""
    status_note, cards_html, trajectory_fig, justification, cpt_badge_html, strategy_html, session_result, patient_id, is_known = \
        _run_pipeline(transcript, patient_id, duration_minutes, diagnosis_choice)

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
        "Medical_Necessity_Justification": justification,
    }
    new_state["session_log"][patient_id] = new_state["session_log"].get(patient_id, []) + [logged_record]

    return status_note, cards_html, trajectory_fig, justification, cpt_badge_html, strategy_html, new_state


def quick_start_analyze(patient_id, transcript, duration_minutes, diagnosis_choice):
    """Used by the Quick Starters — runs the real pipeline live so evaluators
    see it actually work, but doesn't touch session state (a demo run isn't
    meant to permanently add to a clinician's caseload)."""
    status_note, cards_html, trajectory_fig, justification, cpt_badge_html, strategy_html, _, _, _ = \
        _run_pipeline(transcript, patient_id, duration_minutes, diagnosis_choice)
    return status_note, cards_html, trajectory_fig, justification, cpt_badge_html, strategy_html


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
# Layout
# ==============================================================================
CUSTOM_CSS = """
.gradio-container {background: #f4f6f8;}
#nav_tabs > .tab-nav {
    flex-direction: column;
    width: 190px;
    border-right: 1px solid #e0e0e0;
    border-bottom: none;
    gap: 4px;
    padding-top: 8px;
}
#nav_tabs > .tab-nav button {
    text-align: left;
    justify-content: flex-start;
    border-radius: 6px;
    margin: 0 8px;
}
"""

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
        doctor_label = gr.Markdown("")

        with gr.Tabs(elem_id="nav_tabs") as nav_tabs:

            # ---------------- Dashboard ----------------
            with gr.Tab("Dashboard", id="dashboard"):
                gr.Markdown("#### Today's Schedule")
                gr.Markdown(
                    "*NavIO's dataset models historical therapy sessions, not future bookings — "
                    "this schedule is a demo simulation for illustration, distinct from the real "
                    "session archive in each patient's profile.*"
                )
                schedule_df = gr.Dataframe(headers=["Time", "Patient_ID", "Status"], interactive=False)
                dashboard_selection_label = gr.Markdown("")
                dashboard_selected_patient = gr.State(value=None)
                start_session_dash_btn = gr.Button("Start Session with Selected Patient", variant="primary")

            # ---------------- Patients ----------------
            with gr.Tab("Patients", id="patients"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Patient Directory")
                        search_box = gr.Textbox(label="Search by Patient ID", placeholder="e.g. P0001")
                        roster_df = gr.Dataframe(headers=["Patient_ID"], interactive=False)
                        with gr.Accordion("Register New Patient (Intake)", open=False):
                            new_patient_id_box = gr.Textbox(label="New Patient ID", placeholder="e.g. P_NEW_001")
                            register_button = gr.Button("Register")
                            register_error = gr.Markdown(value="", visible=False)
                    with gr.Column(scale=2):
                        profile_header = gr.Markdown("Search or select a patient from the roster on the left.")
                        profile_history_df = gr.Dataframe(
                            headers=["Session #", "Diagnosis", "Risk Score", "Risk Level", "Dynamic", "Progress", "CPT Code", "Crisis"],
                            interactive=False,
                        )
                        profile_session_detail = gr.HTML(value="", visible=False)
                        profile_trajectory_plot = gr.Plot()
                        add_session_profile_btn = gr.Button("Add New Session", variant="primary")

            # ---------------- Sessions (the analysis workspace) ----------------
            with gr.Tab("Sessions", id="sessions"):
                gr.Markdown("#### New Session — Analysis & Documentation")
                gr.Markdown(
                    "*Arriving from Dashboard or a Patient Profile pre-fills the Patient ID below "
                    "(still editable). Visiting this tab directly starts blank — typing an unrecognized "
                    "Patient ID will auto-register it as a new intake patient on submit.*"
                )
                with gr.Row():
                    session_patient_id = gr.Dropdown(
                        label="Patient ID", choices=ALL_PATIENT_IDS, allow_custom_value=True,
                    )
                    session_duration = gr.Number(label="Session Duration (minutes)", value=45, minimum=1)
                    session_diagnosis = gr.Dropdown(label="Primary Diagnosis (override)", choices=DIAGNOSIS_CHOICES, value="Auto-detect")
                session_transcript = gr.Textbox(label="Session Transcript", lines=8, placeholder="Paste or type the session transcript here...")
                analyze_button = gr.Button("Analyze Session", variant="primary")

                # Output components are created with render=False so they can be
                # referenced by gr.Examples' outputs= below (Gradio needs the actual
                # object, not a forward reference), while still visually placing the
                # Quick Starters table above the results via explicit .render() calls.
                session_status = gr.Markdown("", render=False)
                session_cards = gr.HTML(render=False)
                session_trajectory_plot = gr.Plot(label="Risk Trajectory", render=False)
                justification_box = gr.Textbox(label="Medical Necessity Justification", lines=8, interactive=False, render=False)
                cpt_badge = gr.HTML(render=False)
                strategy_output = gr.HTML(label="Next-Session Strategy", render=False)

                # fn+outputs here means clicking an example doesn't just pre-fill the
                # inputs — it runs the live pipeline immediately, satisfying "1-click
                # and see an example of how the application works."
                gr.Examples(
                    examples=build_quick_starters(),
                    inputs=[session_patient_id, session_transcript, session_duration, session_diagnosis],
                    outputs=[session_status, session_cards, session_trajectory_plot, justification_box, cpt_badge, strategy_output],
                    fn=quick_start_analyze,
                    cache_examples=False,
                    label="Quick Starters",
                )

                session_status.render()
                session_cards.render()
                session_trajectory_plot.render()
                with gr.Row():
                    with gr.Column(scale=2):
                        justification_box.render()
                    with gr.Column(scale=1):
                        cpt_badge.render()
                strategy_output.render()

    # ==========================================================================
    # Wiring
    # ==========================================================================
    login_button.click(
        fn=handle_login,
        inputs=[login_name, login_id, app_state],
        outputs=[login_group, main_group, app_state, login_error, doctor_label, schedule_df, roster_df],
    )

    schedule_df.select(fn=on_dashboard_select, inputs=[app_state], outputs=[dashboard_selected_patient, dashboard_selection_label])
    start_session_dash_btn.click(
        fn=start_session_from_dashboard,
        inputs=[dashboard_selected_patient, app_state],
        outputs=[session_patient_id, app_state, nav_tabs],
    )

    search_box.change(fn=filter_roster, inputs=[search_box, app_state], outputs=[roster_df])
    roster_df.select(fn=on_roster_select, inputs=[app_state], outputs=[profile_header, profile_history_df, profile_trajectory_plot, profile_session_detail, app_state])
    profile_history_df.select(fn=on_profile_history_select, inputs=[app_state], outputs=[profile_session_detail])
    register_button.click(
        fn=register_new_patient,
        inputs=[new_patient_id_box, app_state],
        outputs=[roster_df, app_state, register_error, profile_header, profile_history_df, profile_trajectory_plot, profile_session_detail],
    )
    add_session_profile_btn.click(
        fn=add_session_from_profile,
        inputs=[app_state],
        outputs=[session_patient_id, app_state, nav_tabs],
    )

    analyze_button.click(
        fn=run_session_analysis,
        inputs=[session_transcript, session_patient_id, session_duration, session_diagnosis, app_state],
        outputs=[session_status, session_cards, session_trajectory_plot, justification_box, cpt_badge, strategy_output, app_state],
    )


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)
