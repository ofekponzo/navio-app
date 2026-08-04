"""
NavIO — ui_helpers.py
Presentation-layer helpers for app.py: KPI cards, CPT badges, the patient
history table, the risk-trajectory plot, and the dataset/session-log merge.

Deliberately kept separate from recsys.py/generation.py — this file has no
ML logic, only formatting. Note on the dataset schema (confirmed from Part 1):
NavIO's synthetic sessions are numbered 1, 3, 5, 7, 10 (a longitudinal arc
with gaps), not sequential 1-5, and there is no real per-session timestamp
column — history is ordered and displayed by Session_Number, not by date.
Only a NEW session analyzed live in this app has real Session_Start/
Session_End values (derived from "now" in recsys.analyze_new_session()).
"""

import matplotlib
matplotlib.use("Agg")  # headless — no display backend on a Space server
import matplotlib.pyplot as plt
import pandas as pd

RISK_COLORS = {"Low": "#2e7d32", "Medium": "#e6a700", "High": "#c62828"}
CPT_COLORS = {"90832": "#546e7a", "90834": "#1565c0", "90837": "#00695c", "90839": "#c62828"}


def risk_level_color(level):
    return RISK_COLORS.get(level, "#546e7a")


def _card(label, value, accent="#546e7a"):
    return f"""
    <div style="flex:1; min-width:140px; background:#fff; border-left:4px solid {accent};
                border-radius:6px; padding:10px 14px; box-shadow:0 1px 3px rgba(0,0,0,0.08);">
        <div style="font-size:11px; color:#78909c; text-transform:uppercase; letter-spacing:0.04em;">{label}</div>
        <div style="font-size:20px; font-weight:600; color:#263238; margin-top:2px;">{value}</div>
    </div>"""


def format_metric_cards_html(session_result):
    """session_result is recsys.analyze_new_session()'s output dict — has
    Session_Start/Session_End, unlike a raw historical dataset record."""
    risk_color = risk_level_color(session_result["Risk_Level"])
    start = session_result["Session_Start"].strftime("%H:%M")
    end = session_result["Session_End"].strftime("%H:%M")
    progress = session_result.get("Progress", "—")
    progress_accent = "#c62828" if "Mixed Signal" in str(progress) else "#546e7a"

    cards = [
        _card("Session Start / End", f"{start} – {end}"),
        _card("Risk Score", f"{session_result['Predicted_Risk_Score']:.1f} / 10", risk_color),
        _card("Risk Level", session_result["Risk_Level"], risk_color),
        _card("Dynamic", session_result["Predicted_Dynamic"]),
        _card("Progress", progress, progress_accent),
    ]
    return f'<div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px;">{"".join(cards)}</div>'


def format_cpt_badge_html(cpt_code, crisis_flag):
    color = CPT_COLORS.get(cpt_code, "#546e7a")
    label = "CRISIS SESSION" if crisis_flag else "STANDARD SESSION"
    return f"""
    <div style="background:{color}15; border:2px solid {color}; border-radius:10px;
                padding:16px; text-align:center;">
        <div style="font-size:12px; color:{color}; font-weight:600; letter-spacing:0.05em;">{label}</div>
        <div style="font-size:32px; font-weight:700; color:{color}; margin-top:4px;">{cpt_code}</div>
    </div>"""


def format_strategy_html(strategy_text):
    return f"""
    <div style="background:#eef6f3; border-left:5px solid #00695c; border-radius:6px;
                padding:16px 18px; line-height:1.55; color:#1b3a34;">
        {strategy_text}
    </div>"""


def no_history_html():
    return """
    <div style="text-align:center; padding:28px 12px; color:#78909c;">
        <div style="font-size:15px; font-weight:600;">No prior history — Intake session</div>
        <div style="font-size:12px; margin-top:4px;">This patient has no recorded sessions yet.</div>
    </div>"""


def merge_history(dataset_history, session_log_entries):
    """Combines real dataset history (from recsys.retrieval_index) with any
    sessions logged in-memory during this browser session (gr.State), sorted
    by Session_Number. Never mutates the shared, process-wide dataset."""
    combined = list(dataset_history) + list(session_log_entries or [])
    return sorted(combined, key=lambda r: r.get("Session_Number", 0))


def next_session_number(history_records):
    if not history_records:
        return 1
    return max(r.get("Session_Number", 0) for r in history_records) + 1


def format_patient_history_df(history_records):
    if not history_records:
        return pd.DataFrame(columns=["Session #", "Diagnosis", "Risk Score", "Risk Level", "Dynamic", "Progress", "CPT Code", "Crisis"])
    rows = []
    for r in history_records:
        rows.append({
            "Session #": r.get("Session_Number"),
            "Diagnosis": r.get("Primary_Diagnosis", "—"),
            "Risk Score": r.get("Risk_Score", r.get("Predicted_Risk_Score", "—")),
            "Risk Level": r.get("Risk_Level", "—"),
            "Dynamic": r.get("Dynamic", r.get("Predicted_Dynamic", "—")),
            "Progress": r.get("Progress", "—"),
            "CPT Code": r.get("Target_CPT_Code", "—"),
            "Crisis": "Yes" if r.get("Crisis_Flag") else "No",
        })
    return pd.DataFrame(rows)


def format_session_detail_html(record):
    """Used when a clinician clicks a row in the history table / sidebar."""
    justification = record.get("Medical_Necessity_Justification") or "No stored justification text for this session."
    risk_color = risk_level_color(record.get("Risk_Level", ""))
    return f"""
    <div style="background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:14px 16px;">
        <div style="font-weight:600; font-size:14px; margin-bottom:6px;">
            Session #{record.get('Session_Number', '—')} &middot; {record.get('Primary_Diagnosis', '—')}
        </div>
        <div style="font-size:13px; color:#455a64; margin-bottom:8px;">
            Risk Score: <b style="color:{risk_color}">{record.get('Risk_Score', '—')}</b>
            ({record.get('Risk_Level', '—')}) &nbsp;|&nbsp;
            Dynamic: {record.get('Dynamic', '—')} &nbsp;|&nbsp;
            Progress: {record.get('Progress', '—')} &nbsp;|&nbsp;
            CPT: {record.get('Target_CPT_Code', '—')}
            {" &nbsp;|&nbsp; <b style='color:#c62828'>CRISIS</b>" if record.get("Crisis_Flag") else ""}
        </div>
        <div style="font-size:12px; color:#37474f; line-height:1.5; border-top:1px dashed #cfd8dc; padding-top:8px;">
            {justification}
        </div>
    </div>"""


def format_similar_cases_df(similar_cases):
    if not similar_cases:
        return pd.DataFrame(columns=["Diagnosis", "Trajectory", "Progress", "Similarity"])
    rows = [{
        "Diagnosis": c.get("Primary_Diagnosis", "—"),
        "Trajectory": c.get("Trajectory_Type", "—"),
        "Progress": c.get("Progress", "—"),
        "Similarity": round(c.get("similarity", 0), 3),
    } for c in similar_cases]
    return pd.DataFrame(rows)


def build_risk_trajectory_figure(history_records, new_session_result=None):
    """Historical Risk_Score by Session_Number, with crisis sessions marked
    distinctly and (if provided) the just-analyzed new session appended as a
    visually distinct 'in progress' point rather than a recorded history
    point."""
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=110)

    # Risk-level background bands (mirrors bucket_risk_level's thresholds)
    ax.axhspan(0, 3, color=RISK_COLORS["Low"], alpha=0.06)
    ax.axhspan(3, 6, color=RISK_COLORS["Medium"], alpha=0.06)
    ax.axhspan(6, 10, color=RISK_COLORS["High"], alpha=0.06)

    if not history_records and new_session_result is None:
        ax.text(0.5, 0.5, "No trajectory yet — this will be the baseline session",
                 ha="center", va="center", transform=ax.transAxes, color="#78909c", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        return fig

    x_hist = [r.get("Session_Number") for r in history_records]
    y_hist = [r.get("Risk_Score") for r in history_records]
    crisis_hist = [bool(r.get("Crisis_Flag")) for r in history_records]

    if x_hist:
        ax.plot(x_hist, y_hist, color="#37474f", linewidth=1.8, marker="o", markersize=5, zorder=2, label="Recorded sessions")
        for x, y, is_crisis in zip(x_hist, y_hist, crisis_hist):
            if is_crisis:
                ax.scatter([x], [y], color=RISK_COLORS["High"], s=90, zorder=3, marker="X", label="_nolegend_")

    if new_session_result is not None:
        new_x = next_session_number(history_records)
        new_y = new_session_result["Predicted_Risk_Score"]
        if x_hist:
            ax.plot([x_hist[-1], new_x], [y_hist[-1], new_y], color="#8a94a6", linewidth=1.2, linestyle="--", zorder=1)
        star_color = RISK_COLORS["High"] if new_session_result["Crisis_Flag"] else "#1565c0"
        ax.scatter([new_x], [new_y], color=star_color, s=180, marker="*", zorder=4, label="Current session")

    ax.set_xlabel("Session #", fontsize=9)
    ax.set_ylabel("Risk Score", fontsize=9)
    ax.set_ylim(-0.3, 10.3)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    return fig
