"""
NavIO — ui_helpers.py
Presentation-layer helpers for app.py: the design-system palette, KPI cards,
CPT badges, dashboard appointment cards, the patient history table, the two
clinical graphs, and the dataset/session-log merge.

Deliberately kept separate from recsys.py/generation.py — this file has no
ML logic, only formatting. The one exception worth flagging: Graph 2 (the
intra-session timeline) is FED pre-computed per-turn data from
recsys.parse_intrasession_timeline() (timestamp parsing, crisis-keyword
matching, sentiment-based intensity) — this file only lays that data out on
a chart. Keeping the parsing/scoring in recsys.py preserves the same
boundary this module has always had.

Note on the dataset schema (confirmed from Part 1): NavIO's synthetic
sessions are numbered 1, 3, 5, 7, 10 (a longitudinal arc with gaps), not
sequential 1-5, and there is no real per-session timestamp column — history
is ordered and displayed by Session_Number, not by date. Only a NEW session
analyzed live in this app has real Session_Start/Session_End values
(derived from "now" in recsys.analyze_new_session()).

All UI copy is English-only by project requirement, regardless of the
language used in conversation with the assistant that built this.
"""

import matplotlib
matplotlib.use("Agg")  # headless — no display backend on a Space server
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# ==============================================================================
# Design-system palette — shared with app.py's CUSTOM_CSS so Python-generated
# HTML (cards, badges) and pure-CSS-styled Gradio components read as one
# consistent system. Palette + type choices are deliberately matched to the
# reference enterprise-EMR mockup the project is styled after: a dark navy
# sidebar, a teal primary accent, and IBM Plex Mono reserved for uppercase
# labels / data figures (Hanken Grotesk carries all body text).
# ==============================================================================
COLOR_BG = "#f5f7fb"
COLOR_CARD = "#ffffff"
COLOR_BORDER = "#e7edf4"
COLOR_TEXT_PRIMARY = "#14223b"
COLOR_TEXT_SECONDARY = "#6b7b90"
COLOR_TEXT_MUTED = "#9aa7b8"

COLOR_ACCENT = "#06b4ab"        # teal — primary actions, positive/synced status
COLOR_ACCENT_HOVER = "#0a857e"
COLOR_ACCENT_SOFT = "#e6f7f5"
COLOR_SECONDARY = "#3b6fe0"     # blue — secondary data accent
COLOR_SECONDARY_SOFT = "#eef4ff"
COLOR_SUCCESS = "#10b981"
COLOR_SUCCESS_SOFT = "#eafaf0"
COLOR_SUCCESS_TEXT = "#0c7a4f"
COLOR_WARNING = "#f59e0b"
COLOR_WARNING_SOFT = "#fdf2dd"
COLOR_WARNING_TEXT = "#b4791a"
COLOR_DANGER = "#ef5b5b"
COLOR_DANGER_SOFT = "#fdeceb"
COLOR_DANGER_TEXT = "#c23636"
COLOR_SIDEBAR_BG = "#0f1b30"
COLOR_SIDEBAR_BORDER = "#1c2c49"

FONT_BODY = "'Hanken Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

RISK_COLORS = {"Low": COLOR_SUCCESS_TEXT, "Medium": COLOR_WARNING_TEXT, "High": COLOR_DANGER_TEXT}
RISK_SOFT = {"Low": COLOR_SUCCESS_SOFT, "Medium": COLOR_WARNING_SOFT, "High": COLOR_DANGER_SOFT}
RISK_LINE = {"Low": COLOR_SUCCESS, "Medium": COLOR_WARNING, "High": COLOR_DANGER}

CPT_COLORS = {"90832": "#546e7a", "90834": COLOR_SECONDARY, "90837": COLOR_ACCENT, "90839": COLOR_DANGER_TEXT}
CPT_SOFT = {"90832": "#eceff1", "90834": COLOR_SECONDARY_SOFT, "90837": COLOR_ACCENT_SOFT, "90839": COLOR_DANGER_SOFT}
STATUS_COLORS = {"Scheduled": (COLOR_SECONDARY, COLOR_SECONDARY_SOFT), "Completed": (COLOR_SUCCESS_TEXT, COLOR_SUCCESS_SOFT)}


def risk_level_color(level):
    return RISK_COLORS.get(level, COLOR_TEXT_SECONDARY)


def _mono_label(text, color=COLOR_TEXT_SECONDARY):
    return (
        f'<span style="font-family:{FONT_MONO}; font-size:9.5px; letter-spacing:0.1em; '
        f'text-transform:uppercase; font-weight:600; color:{color};">{text}</span>'
    )


# ==============================================================================
# KPI / metric cards
# ==============================================================================
def _card(label, value, accent=COLOR_ACCENT):
    return f"""
    <div style="flex:1; min-width:150px; background:{COLOR_CARD}; border:1px solid {COLOR_BORDER};
                border-radius:14px; padding:15px 17px; box-shadow:0 4px 14px rgba(16,34,59,0.05);
                border-top:3px solid {accent};">
        <div style="font-family:{FONT_MONO}; font-size:10px; color:{COLOR_TEXT_MUTED}; text-transform:uppercase;
                    letter-spacing:0.09em; font-weight:600;">{label}</div>
        <div style="font-family:{FONT_BODY}; font-size:22px; font-weight:800; color:{COLOR_TEXT_PRIMARY};
                    margin-top:5px; letter-spacing:-0.01em;">{value}</div>
    </div>"""


def format_metric_cards_html(session_result):
    """session_result is recsys.analyze_new_session()'s output dict — has
    Session_Start/Session_End, unlike a raw historical dataset record."""
    risk_color = risk_level_color(session_result["Risk_Level"])
    start = session_result["Session_Start"].strftime("%H:%M")
    end = session_result["Session_End"].strftime("%H:%M")
    progress = session_result.get("Progress", "—")
    progress_accent = COLOR_DANGER_TEXT if "Mixed Signal" in str(progress) else COLOR_ACCENT

    cards = [
        _card("Session Start / End", f"{start} – {end}", COLOR_SECONDARY),
        _card("Risk Score", f"{session_result['Predicted_Risk_Score']:.1f} / 10", risk_color),
        _card("Risk Level", session_result["Risk_Level"], risk_color),
        _card("Dynamic", session_result["Predicted_Dynamic"]),
        _card("Progress", progress, progress_accent),
    ]
    return f'<div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:4px;">{"".join(cards)}</div>'


def format_cpt_badge_html(cpt_code, crisis_flag):
    color = CPT_COLORS.get(cpt_code, COLOR_TEXT_SECONDARY)
    soft = CPT_SOFT.get(cpt_code, COLOR_BG)
    label = "CRISIS SESSION" if crisis_flag else "STANDARD SESSION"
    label_color = COLOR_DANGER_TEXT if crisis_flag else COLOR_TEXT_MUTED
    return f"""
    <div style="background:{soft}; border:1px solid {color}33; border-radius:16px;
                padding:22px 18px; text-align:center; height:100%; box-sizing:border-box;
                display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px;">
        {_mono_label(label, label_color)}
        <div style="font-family:{FONT_MONO}; font-size:38px; font-weight:700; color:{color};
                    letter-spacing:-0.02em;">{cpt_code}</div>
        <div style="font-size:11.5px; color:{COLOR_TEXT_MUTED};">Recommended CPT billing code</div>
    </div>"""


_STRATEGY_ICON_SVG = (
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>'
    '<circle cx="12" cy="12" r="4"></circle></svg>'
)


def format_strategy_html(strategy_text):
    """Deliberately the most visually prominent container on the Sessions
    page — a distinct, soft-accent card rather than a plain textbox — since
    this is the clinician-facing recommendation the whole pipeline builds
    toward. The 'clinical judgment required' caption is a direct, visible
    reminder that this is decision SUPPORT, not an autonomous instruction,
    consistent with the hallucination-guard framing used in generation.py."""
    icon = _STRATEGY_ICON_SVG.format(color=COLOR_ACCENT)
    return f"""
    <div style="background:linear-gradient(135deg, {COLOR_ACCENT_SOFT}, #ffffff 65%);
                border:1px solid {COLOR_ACCENT}40; border-left:4px solid {COLOR_ACCENT};
                border-radius:16px; padding:22px 24px; margin-top:6px;
                box-shadow:0 8px 22px rgba(6,180,171,0.08);">
        <div style="display:flex; align-items:center; justify-content:space-between; gap:14px;
                    flex-wrap:wrap; margin-bottom:14px;">
            <div style="display:flex; align-items:center; gap:11px;">
                <div style="width:34px; height:34px; border-radius:10px; background:{COLOR_ACCENT_SOFT};
                            display:flex; align-items:center; justify-content:center; flex-shrink:0;">{icon}</div>
                <div style="font-family:{FONT_BODY}; font-size:15.5px; font-weight:800; color:{COLOR_TEXT_PRIMARY};
                            letter-spacing:-0.01em;">Next-Session Strategy</div>
            </div>
            {_mono_label("Clinical judgment required", COLOR_TEXT_MUTED)}
        </div>
        <div style="font-family:{FONT_BODY}; line-height:1.65; color:#0f3d38; font-size:14.5px;">
            {strategy_text}
        </div>
    </div>"""


def no_history_html():
    return f"""
    <div style="text-align:center; padding:34px 12px; color:{COLOR_TEXT_MUTED};
                background:{COLOR_CARD}; border:1px dashed {COLOR_BORDER}; border-radius:14px;">
        <div style="font-size:15px; font-weight:700; color:{COLOR_TEXT_SECONDARY};">No prior history — Intake session</div>
        <div style="font-size:12px; margin-top:4px;">This patient has no recorded sessions yet.</div>
    </div>"""


# ==============================================================================
# Dashboard appointment cards (replaces a raw gr.Dataframe schedule listing)
# ==============================================================================
def format_schedule_card_html(entry):
    status = entry.get("Status", "Scheduled")
    status_color, status_soft = STATUS_COLORS.get(status, (COLOR_TEXT_SECONDARY, COLOR_BG))
    return f"""
    <div style="display:flex; align-items:center; justify-content:space-between; gap:16px;
                background:{COLOR_CARD}; border:1px solid {COLOR_BORDER}; border-radius:14px;
                padding:15px 19px; box-shadow:0 3px 10px rgba(16,34,59,0.04);">
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="font-family:{FONT_MONO}; font-size:13.5px; font-weight:700; color:{COLOR_TEXT_PRIMARY};
                        min-width:56px;">{entry.get('Time', '—')}</div>
            <div>
                <div style="font-size:14px; font-weight:700; color:{COLOR_TEXT_PRIMARY};">{entry.get('Patient_ID', '—')}</div>
                <div style="font-size:11px; color:{COLOR_TEXT_MUTED}; margin-top:1px;">Outpatient psychotherapy</div>
            </div>
        </div>
        <div style="background:{status_soft}; color:{status_color}; font-family:{FONT_MONO}; font-size:10.5px;
                    font-weight:700; padding:5px 11px; border-radius:999px; letter-spacing:0.05em;">{status.upper()}</div>
    </div>"""


def no_schedule_html():
    return f"""
    <div style="text-align:center; padding:30px 12px; color:{COLOR_TEXT_MUTED};
                background:{COLOR_CARD}; border:1px dashed {COLOR_BORDER}; border-radius:14px;">
        No appointments on today's schedule.
    </div>"""


# ==============================================================================
# Patient history / merge helpers
# ==============================================================================
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
    """Used when a clinician clicks a row in the history table."""
    justification = record.get("Medical_Necessity_Justification") or "No stored justification text for this session."
    risk_color = risk_level_color(record.get("Risk_Level", ""))
    crisis_chip = (
        f"<span style='color:{COLOR_DANGER_TEXT}; font-weight:700;'>CRISIS</span> &nbsp;|&nbsp; "
        if record.get("Crisis_Flag") else ""
    )
    return f"""
    <div style="background:{COLOR_CARD}; border:1px solid {COLOR_BORDER}; border-radius:14px; padding:17px 19px;
                box-shadow:0 3px 10px rgba(16,34,59,0.04);">
        <div style="font-weight:700; font-size:14px; margin-bottom:8px; color:{COLOR_TEXT_PRIMARY};">
            Session #{record.get('Session_Number', '—')} &middot; {record.get('Primary_Diagnosis', '—')}
        </div>
        <div style="font-size:12.5px; color:{COLOR_TEXT_SECONDARY}; margin-bottom:10px;">
            {crisis_chip}
            Risk Score: <b style="color:{risk_color}">{record.get('Risk_Score', '—')}</b>
            ({record.get('Risk_Level', '—')}) &nbsp;|&nbsp;
            Dynamic: {record.get('Dynamic', '—')} &nbsp;|&nbsp;
            Progress: {record.get('Progress', '—')} &nbsp;|&nbsp;
            CPT: {record.get('Target_CPT_Code', '—')}
        </div>
        <div style="font-size:12.5px; color:{COLOR_TEXT_SECONDARY}; line-height:1.55; border-top:1px dashed {COLOR_BORDER}; padding-top:10px;">
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


# ==============================================================================
# Shared matplotlib styling — clean clinical look: transparent background, no
# default chart-junk (heavy borders/ticks), muted horizontal-only gridlines.
# ==============================================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hanken Grotesk", "DejaVu Sans", "Arial", "Helvetica"]


def _style_axes(fig, ax):
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.tick_params(labelsize=8, colors=COLOR_TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=COLOR_BORDER, linewidth=0.8, zorder=0)
    for spine_name, spine in ax.spines.items():
        spine.set_visible(spine_name == "bottom")
        if spine_name == "bottom":
            spine.set_color(COLOR_BORDER)


def _empty_figure(message):
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=120)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
             color=COLOR_TEXT_MUTED, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


# ==============================================================================
# GRAPH 1 — Cross-Session Progress & Trajectory. X: Session Number, Y: 0-10
# Risk_Score. Historical scores plus (if provided) the just-analyzed new
# session's predicted score as a distinct 'in progress' point.
# ==============================================================================
def build_risk_trajectory_figure(history_records, new_session_result=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=120)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Risk-level background bands (mirrors bucket_risk_level's thresholds)
    ax.axhspan(0, 3, color=RISK_LINE["Low"], alpha=0.05, zorder=0)
    ax.axhspan(3, 6, color=RISK_LINE["Medium"], alpha=0.05, zorder=0)
    ax.axhspan(6, 10, color=RISK_LINE["High"], alpha=0.05, zorder=0)

    if not history_records and new_session_result is None:
        return _empty_figure("No trajectory yet — this will be the baseline session")

    x_hist = [r.get("Session_Number") for r in history_records]
    y_hist = [r.get("Risk_Score") for r in history_records]
    crisis_hist = [bool(r.get("Crisis_Flag")) for r in history_records]

    if x_hist:
        ax.plot(x_hist, y_hist, color=COLOR_ACCENT, linewidth=2.2, marker="o",
                markersize=5.5, markerfacecolor="white", markeredgecolor=COLOR_ACCENT,
                markeredgewidth=1.8, zorder=2, label="Recorded sessions")
        for x, y, is_crisis in zip(x_hist, y_hist, crisis_hist):
            if is_crisis:
                ax.scatter([x], [y], color=COLOR_DANGER, s=110, zorder=3, marker="X", label="_nolegend_")

    if new_session_result is not None:
        new_x = next_session_number(history_records)
        new_y = new_session_result["Predicted_Risk_Score"]
        if x_hist:
            ax.plot([x_hist[-1], new_x], [y_hist[-1], new_y], color=COLOR_TEXT_MUTED,
                     linewidth=1.4, linestyle="--", zorder=1)
        star_color = COLOR_DANGER if new_session_result["Crisis_Flag"] else COLOR_ACCENT
        ax.scatter([new_x], [new_y], color=star_color, s=220, marker="*", zorder=4,
                   edgecolors="white", linewidths=0.8, label="Current session")

    ax.set_xlabel("Session #", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("Risk Score", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    ax.set_ylim(-0.3, 10.3)
    _style_axes(fig, ax)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=COLOR_TEXT_SECONDARY)
    fig.tight_layout()
    return fig


# ==============================================================================
# GRAPH 2 — Intra-Session Clinical Timeline. X: minutes 0 -> session duration.
# Plots a per-turn "intensity" line (from recsys.parse_intrasession_timeline)
# with Safe / Alert / Flooding zone bands, colors each turn marker by speaker
# role (communication shifts), and calls out crisis-keyword turns explicitly.
# ==============================================================================
def build_intrasession_timeline_figure(timeline_turns, duration_minutes, crisis_flag=False):
    if not timeline_turns:
        return _empty_figure("No [MM:SS] timestamps found in this transcript —\nGraph 2 requires timestamped turns, e.g. \"[00:12] Dr. Carter: ...\"")

    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=120)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    x_max = max(float(duration_minutes or 0), max(t["time_min"] for t in timeline_turns)) + 1.0

    # Safe / Alert / Flooding zone bands (matches the clinical language used
    # elsewhere in the project's reference materials for arousal/regulation).
    ax.axhspan(0, 40, color=COLOR_SUCCESS, alpha=0.06, zorder=0)
    ax.axhspan(40, 70, color=COLOR_WARNING, alpha=0.07, zorder=0)
    ax.axhspan(70, 100, color=COLOR_DANGER, alpha=0.07, zorder=0)
    for y, txt in [(20, "Safe"), (55, "Alert"), (85, "Flooding")]:
        ax.text(x_max, y, txt, fontsize=7, color=COLOR_TEXT_MUTED, ha="right", va="center",
                 style="italic", zorder=1)

    xs = [t["time_min"] for t in timeline_turns]
    ys = [t["intensity"] for t in timeline_turns]

    ax.plot(xs, ys, color=COLOR_SECONDARY, linewidth=1.8, zorder=2, alpha=0.85)

    for t in timeline_turns:
        is_crisis = bool(t["crisis_hits"])
        role_color = COLOR_ACCENT if t["role"] == "patient" else "#8a94a6"
        marker = "X" if is_crisis else "o"
        size = 130 if is_crisis else 46
        face = COLOR_DANGER if is_crisis else role_color
        ax.scatter([t["time_min"]], [t["intensity"]], s=size, marker=marker, color=face,
                   edgecolors="white", linewidths=1.0, zorder=4 if is_crisis else 3)

    # Call out the single highest-intensity crisis-keyword turn, if any —
    # mirrors how a clinician would want the peak moment labeled directly.
    crisis_turns = [t for t in timeline_turns if t["crisis_hits"]]
    if crisis_turns:
        peak = max(crisis_turns, key=lambda t: t["intensity"])
        ax.axvline(peak["time_min"], color=COLOR_DANGER, linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
        minute, second = int(peak["time_min"]), int(round((peak["time_min"] % 1) * 60))
        ax.annotate(f"{minute:02d}:{second:02d}", xy=(peak["time_min"], peak["intensity"]),
                    xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=7.5, fontweight="bold", color=COLOR_DANGER_TEXT)

    # Legend proxies (role + crisis marker) — built manually since the real
    # scatter calls are per-point, not per-series.
    import matplotlib.lines as mlines
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_ACCENT, marker="o", linestyle="None", markersize=6, label="Patient turn"),
        mlines.Line2D([], [], color="#8a94a6", marker="o", linestyle="None", markersize=6, label="Therapist turn"),
    ]
    if crisis_turns:
        legend_handles.append(
            mlines.Line2D([], [], color=COLOR_DANGER, marker="X", linestyle="None", markersize=8, label="Crisis keyword")
        )

    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-4, 108)
    ax.set_xlabel("Session Time (minutes)", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("Estimated Intensity", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    _style_axes(fig, ax)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5, frameon=False,
              labelcolor=COLOR_TEXT_SECONDARY, ncol=len(legend_handles))
    fig.tight_layout()
    return fig
