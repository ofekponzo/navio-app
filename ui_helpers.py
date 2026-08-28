"""
NavIO — ui_helpers.py
Presentation-layer helpers for app.py: the design-system palette, KPI/ring
cards, CPT badges, the calendar-style dashboard schedule, the patient history
table, both clinical graphs, and the dataset/session-log merge.
Deliberately kept separate from recsys.py/generation.py — this file has no
ML logic, only formatting. The one exception worth flagging: Graph 2 (the
intra-session timeline) is FED pre-computed per-turn data from
recsys.parse_intrasession_timeline() (timestamp parsing, crisis-keyword
matching, sentiment-based intensity) — this file only lays that data out on
a chart (including the proportional time-rescaling and annotation logic,
which are chart-presentation decisions, not model outputs).
Note on the dataset schema (confirmed from Part 1): NavIO's synthetic
sessions are numbered 1, 3, 5, 7, 10 (a longitudinal arc with gaps), not
sequential 1-5, and there is no real per-session timestamp column — history
is ordered and displayed by Session_Number, not by date. Only a NEW session
analyzed live in this app has real Session_Start/Session_End values
(derived from "now" in recsys.analyze_new_session()). This module is
written to accept EITHER a live session_result dict (Predicted_Risk_Score,
Predicted_Dynamic, Session_Start/End, Crisis_Probability, ...) OR a stored
historical record dict (Risk_Score, Dynamic, Session_Number, ...) via the
`_first()` key-agnostic lookup helper below — this is what lets the
Patients page render a past session's FULL dashboard (metrics, both
graphs, CPT badge, justification, strategy) inline, with no popup/modal.
All UI copy is English-only by project requirement, regardless of the
language used in conversation with the assistant that built this.
"""
import re
import matplotlib
matplotlib.use("Agg")  # headless — no display backend on a Space server
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
# ==============================================================================
# Design-system palette — shared with app.py's CUSTOM_CSS so Python-generated
# HTML (cards, badges, calendar) and pure-CSS-styled Gradio components read as
# one consistent system. Palette + type choices are matched to the light,
# whitespace-heavy clinical-SaaS aesthetic the app is styled after: pure-white
# elevated cards on a very light off-white ground, a teal primary accent used
# for actions / active states / key data points ONLY, and IBM Plex Mono
# reserved for uppercase labels / data figures (Hanken Grotesk carries all
# body text). green/amber/red are reserved for clinical STATE — never the
# accent. Every value below mirrors a --navio-* CSS variable in app.py.
# ==============================================================================
COLOR_BG = "#f6f8fb"
COLOR_CARD = "#ffffff"
COLOR_BORDER = "#e8edf3"
COLOR_TEXT_PRIMARY = "#16233d"
COLOR_TEXT_SECONDARY = "#5c6b81"
COLOR_TEXT_MUTED = "#93a1b4"
COLOR_ACCENT = "#0ea5a9"        # teal — primary actions, positive/synced status
COLOR_ACCENT_HOVER = "#0c8f93"
COLOR_ACCENT_SOFT = "#e6f6f6"
COLOR_SECONDARY = "#3b6fe0"     # blue — secondary data accent
COLOR_SECONDARY_SOFT = "#eef4ff"
COLOR_SUCCESS = "#17a673"
COLOR_SUCCESS_SOFT = "#e6f6ef"
COLOR_SUCCESS_TEXT = "#0c7a4f"
COLOR_WARNING = "#e08a1e"
COLOR_WARNING_SOFT = "#fdf1e0"
COLOR_WARNING_TEXT = "#b4791a"
COLOR_DANGER = "#e0483c"
COLOR_DANGER_SOFT = "#fdeceb"
COLOR_DANGER_TEXT = "#c23636"
COLOR_SIDEBAR_BG = "#ffffff"    # sidebar is now LIGHT (styled in app.py CSS); kept for reference
COLOR_SIDEBAR_BORDER = "#e8edf3"
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
def _first(d, *keys, default=None):
    """Key-agnostic lookup: a live session_result dict and a stored
    historical record dict use different key names for the same concept
    (Predicted_Risk_Score vs Risk_Score, etc.) — this lets one formatting
    function serve both without the caller normalizing first."""
    for key in keys:
        val = d.get(key)
        if val is not None:
            return val
    return default
# ==============================================================================
# Progress rings — small inline SVG radial indicators for numeric metrics
# (Risk Score, Crisis Probability). Pure SVG/CSS, no JS, safe inside gr.HTML.
# ==============================================================================
def _progress_ring_svg(pct, color, size=62, stroke=7):
    pct = max(0.0, min(100.0, float(pct)))
    radius = (size - stroke) / 2
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - pct / 100.0)
    center = size / 2
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform:rotate(-90deg); flex-shrink:0;">
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{COLOR_BORDER}" stroke-width="{stroke}"/>
        <circle cx="{center}" cy="{center}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke}"
                stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" stroke-linecap="round"/>
    </svg>"""
def _ring_card(label, pct, display_value, color):
    ring = _progress_ring_svg(pct, color)
    return f"""
    <div style="flex:1; min-width:150px; background:{COLOR_CARD}; border:1px solid {COLOR_BORDER};
                border-radius:14px; padding:13px 16px; box-shadow:0 4px 14px rgba(16,34,59,0.05);
                display:flex; align-items:center; gap:12px;">
        <div style="position:relative; width:62px; height:62px; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
            {ring}
            <div style="position:absolute; font-size:14px; font-weight:800; color:{COLOR_TEXT_PRIMARY};">{display_value}</div>
        </div>
        <div style="font-family:{FONT_MONO}; font-size:10px; color:{COLOR_TEXT_MUTED}; text-transform:uppercase;
                    letter-spacing:0.08em; font-weight:600; line-height:1.4;">{label}</div>
    </div>"""
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
def format_metric_cards_html(session_like):
    """Accepts EITHER recsys.analyze_new_session()'s live output dict OR a
    stored historical record dict (dataset row or this-session's logged
    record) — see module docstring. Risk Score and Crisis Probability (when
    present) render as progress rings; Risk Level/Dynamic/Progress stay as
    plain cards since they're categorical, not a 0-100 quantity."""
    risk_score = _first(session_like, "Predicted_Risk_Score", "Risk_Score")
    risk_level = _first(session_like, "Risk_Level", default="—")
    dynamic = _first(session_like, "Predicted_Dynamic", "Dynamic", default="—")
    progress = _first(session_like, "Progress", default="—")
    crisis_prob = session_like.get("Crisis_Probability")
    risk_color = risk_level_color(risk_level)
    progress_accent = COLOR_DANGER_TEXT if "Mixed Signal" in str(progress) else COLOR_ACCENT
    if session_like.get("Session_Start") and session_like.get("Session_End"):
        id_card = _card(
            "Session Start / End",
            f"{session_like['Session_Start'].strftime('%H:%M')} – {session_like['Session_End'].strftime('%H:%M')}",
            COLOR_SECONDARY,
        )
    else:
        id_card = _card("Session #", str(session_like.get("Session_Number", "—")), COLOR_SECONDARY)
    cards = [id_card]
    if risk_score is not None:
        cards.append(_ring_card("Risk Score / 10", float(risk_score) * 10, f"{float(risk_score):.1f}", risk_color))
    cards.append(_card("Risk Level", risk_level, risk_color))
    if crisis_prob is not None:
        ring_color = COLOR_DANGER_TEXT if float(crisis_prob) >= 0.2 else COLOR_ACCENT
        cards.append(_ring_card("Crisis Probability", float(crisis_prob) * 100, f"{float(crisis_prob) * 100:.0f}%", ring_color))
    cards.append(_card("Dynamic", dynamic))
    cards.append(_card("Progress", progress, progress_accent))
    return f'<div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:4px;">{"".join(cards)}</div>'
def format_cpt_badge_html(cpt_code, crisis_flag, duration_minutes=None):
    color = CPT_COLORS.get(cpt_code, COLOR_TEXT_SECONDARY)
    soft = CPT_SOFT.get(cpt_code, COLOR_BG)
    label = "CRISIS SESSION" if crisis_flag else "STANDARD SESSION"
    label_color = COLOR_DANGER_TEXT if crisis_flag else COLOR_TEXT_MUTED
    duration_html = ""
    if duration_minutes:
        duration_html = (
            f'<div style="font-size:12.5px; font-weight:600; color:{COLOR_TEXT_SECONDARY}; '
            f'margin-top:-2px;">{int(duration_minutes)} min face-to-face</div>'
        )
    return f"""
    <div style="background:{soft}; border:1px solid {color}33; border-radius:16px;
                padding:22px 18px; text-align:center; height:100%; box-sizing:border-box;
                display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px;">
        {_mono_label(label, label_color)}
        <div style="font-family:{FONT_MONO}; font-size:38px; font-weight:700; color:{color};
                    letter-spacing:-0.02em;">{cpt_code}</div>
        {duration_html}
        <div style="font-size:11.5px; color:{COLOR_TEXT_MUTED};">Recommended CPT billing code</div>
    </div>"""
_STRATEGY_ICON_SVG = (
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"></path>'
    '<circle cx="12" cy="12" r="4"></circle></svg>'
)
def format_strategy_html(strategy_text):
    """Deliberately the most visually prominent container wherever it
    appears — a distinct, soft-accent card rather than a plain textbox —
    since this is the clinician-facing recommendation the whole pipeline
    builds toward. The 'clinical judgment required' caption is a direct,
    visible reminder that this is decision SUPPORT, not an autonomous
    instruction, consistent with the hallucination-guard framing used in
    generation.py."""
    icon = _STRATEGY_ICON_SVG.format(color=COLOR_ACCENT)
    return f"""
    <div style="background:linear-gradient(135deg, {COLOR_ACCENT_SOFT}, #ffffff 65%);
                border:1px solid {COLOR_ACCENT}40; border-left:4px solid {COLOR_ACCENT};
                border-radius:16px; padding:22px 24px; margin-top:6px;
                box-shadow:0 8px 22px rgba(14,165,169,0.08);">
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
def no_session_selected_html():
    return f"""
    <div style="text-align:center; padding:36px 12px; color:{COLOR_TEXT_MUTED};
                background:{COLOR_CARD}; border:1px dashed {COLOR_BORDER}; border-radius:14px;">
        Select a session from the history table (or the "Select Session to Inspect" dropdown) above
        to view its full record — executive summary, intra-session timeline, CPT code, justification,
        and strategy — right here on this page.
    </div>"""
# ==============================================================================
# Patient Summary Header — shown once a patient is selected from the
# top-level Dropdown, before drilling into any individual session: MRN,
# primary diagnosis, session count, and overall care-episode trajectory.
# ==============================================================================
def format_patient_summary_html(patient_id, history):
    if not patient_id:
        return f"""
        <div style="text-align:center; padding:28px 12px; color:{COLOR_TEXT_MUTED};
                    background:{COLOR_CARD}; border:1px dashed {COLOR_BORDER}; border-radius:14px;">
            Select a patient from the dropdown above to view their care episode summary.
        </div>"""
    if not history:
        cards = [
            _card("MRN", patient_id, COLOR_SECONDARY),
            _card("Total Sessions", "0", COLOR_SECONDARY),
            _card("Status", "Intake — no recorded sessions yet", COLOR_TEXT_SECONDARY),
        ]
        return f'<div style="display:flex; gap:14px; flex-wrap:wrap;">{"".join(cards)}</div>'
    latest, first = history[-1], history[0]
    diagnosis = latest.get("Primary_Diagnosis") or "—"
    latest_score, first_score = latest.get("Risk_Score"), first.get("Risk_Score")
    progress = latest.get("Progress", "—")
    risk_level = latest.get("Risk_Level", "—")
    risk_color = risk_level_color(risk_level)
    cards = [
        _card("MRN", patient_id, COLOR_SECONDARY),
        _card("Primary Diagnosis", diagnosis),
        _card("Total Sessions", str(len(history)), COLOR_SECONDARY),
    ]
    if latest_score is not None:
        cards.append(_ring_card("Current Risk Score", float(latest_score) * 10, f"{float(latest_score):.1f}", risk_color))
    if latest_score is not None and first_score is not None and len(history) > 1:
        delta = float(latest_score) - float(first_score)
        trend_color = COLOR_SUCCESS_TEXT if delta < -0.5 else (COLOR_DANGER_TEXT if delta > 0.5 else COLOR_TEXT_SECONDARY)
        arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "→")
        cards.append(_card("Risk Trajectory", f"{first_score:.1f} {arrow} {latest_score:.1f}", trend_color))
    cards.append(_card("Care Episode Progress", progress, COLOR_DANGER_TEXT if "Mixed Signal" in str(progress) else COLOR_ACCENT))
    return f'<div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:4px;">{"".join(cards)}</div>'
# ==============================================================================
# Session Executive Summary — a deterministic, template-built recap of ONE
# historical session, assembled entirely from already-stored fields (plus,
# if the transcript was retained, a literal quoted opening line). No model
# call, no synthesized claims — every fact here is either a stored field or
# a direct transcript quote, consistent with the project's hallucination-
# guard posture (see generation.py) and the "zero re-inference" requirement
# for browsing historical sessions.
# ==============================================================================
_LEADING_TIMESTAMP_SPEAKER = re.compile(r"^\s*\[\d{1,2}:\d{2}\]\s*[A-Za-z][A-Za-z .]{0,25}:\s*")
def _opening_line(transcript):
    if not transcript:
        return None
    for line in transcript.strip().split("\n"):
        line = line.strip()
        if line:
            return _LEADING_TIMESTAMP_SPEAKER.sub("", line).strip()
    return None
def format_session_summary_html(record):
    session_num = record.get("Session_Number", "—")
    diagnosis = record.get("Primary_Diagnosis") or None
    risk_score = record.get("Risk_Score")
    risk_level = record.get("Risk_Level", "—")
    dynamic = record.get("Dynamic", "—")
    progress = record.get("Progress", "—")
    crisis = bool(record.get("Crisis_Flag"))
    cpt = record.get("Target_CPT_Code", "—")
    risk_color = risk_level_color(risk_level)
    score_text = f"{float(risk_score):.1f}/10" if risk_score is not None else "—"
    sentence = (
        f"Session #{session_num}" + (f" ({diagnosis})" if diagnosis else "") +
        f" recorded a Risk Score of <b style='color:{risk_color};'>{score_text}</b> ({risk_level}), "
        f"with a {dynamic} interaction dynamic. Progress for this session was classified as "
        f"&ldquo;{progress}&rdquo;, billed under CPT {cpt}."
    )
    if crisis:
        sentence += f" <b style='color:{COLOR_DANGER_TEXT};'>This session was flagged for crisis-level clinical content.</b>"
    opening = _opening_line(record.get("Transcript"))
    opening_html = ""
    if opening:
        excerpt = opening if len(opening) <= 160 else opening[:157] + "…"
        opening_html = (
            f'<div style="margin-top:10px; padding-top:10px; border-top:1px dashed {COLOR_BORDER}; '
            f'font-style:italic; color:{COLOR_TEXT_SECONDARY};">Session opened: &ldquo;{excerpt}&rdquo;</div>'
        )
    return f"""
    <div style="background:{COLOR_CARD}; border:1px solid {COLOR_BORDER}; border-radius:14px; padding:17px 19px;
                box-shadow:0 3px 10px rgba(16,34,59,0.04);">
        {_mono_label("Session Executive Summary", COLOR_TEXT_MUTED)}
        <div style="margin-top:8px; font-size:13.5px; line-height:1.6; color:{COLOR_TEXT_PRIMARY};">{sentence}</div>
        {opening_html}
    </div>"""
# ==============================================================================
# Dashboard — calendar-style day view. A purely decorative, self-contained
# hour-ruler visualization (one fully-owned HTML block: no per-component
# absolute positioning, no assumptions about Gradio's own DOM wrapping, so
# it renders reliably regardless of Gradio version) gives the "day at a
# glance" calendar look. Real 1-click entry into a session is handled by
# separate, ordinary gr.Button components below it — Gradio cannot bind a
# Python callback to an arbitrary HTML element, so the interactive control
# has to be a real component, not a click handler baked into this markup.
# ==============================================================================
CALENDAR_START_HOUR = 8
CALENDAR_END_HOUR = 17
CALENDAR_PX_PER_HOUR = 60
CALENDAR_EVENT_MINUTES = 40  # visual block length — the mock schedule has start times only, no explicit duration
def _parse_slot_time(time_str):
    try:
        hour_str, minute_str = str(time_str).split(":")
        return int(hour_str), int(minute_str)
    except (ValueError, AttributeError):
        return None
def build_calendar_strip_html(schedule):
    """One self-contained HTML/CSS visualization of the day: hour gridlines
    + labels, and a colored, time-positioned chip per appointment."""
    total_minutes = (CALENDAR_END_HOUR - CALENDAR_START_HOUR) * 60
    height_px = (CALENDAR_END_HOUR - CALENDAR_START_HOUR) * CALENDAR_PX_PER_HOUR
    rails = []
    for hour in range(CALENDAR_START_HOUR, CALENDAR_END_HOUR + 1):
        top = (hour - CALENDAR_START_HOUR) * CALENDAR_PX_PER_HOUR
        rails.append(
            f'<div style="position:absolute; top:{top}px; left:0; right:0; border-top:1px solid {COLOR_BORDER};"></div>'
            f'<div style="position:absolute; top:{top - 7}px; left:0; font-family:{FONT_MONO}; font-size:10px; '
            f'color:{COLOR_TEXT_MUTED};">{hour:02d}:00</div>'
        )
    chips = []
    for entry in schedule:
        parsed = _parse_slot_time(entry.get("Time", ""))
        if parsed is None:
            continue
        hour, minute = parsed
        start_offset_min = (hour - CALENDAR_START_HOUR) * 60 + minute
        if not (0 <= start_offset_min <= total_minutes):
            continue
        top_px = start_offset_min / 60.0 * CALENDAR_PX_PER_HOUR
        chip_height_px = max(30.0, CALENDAR_EVENT_MINUTES / 60.0 * CALENDAR_PX_PER_HOUR - 4)
        status = entry.get("Status", "Scheduled")
        status_color, status_soft = STATUS_COLORS.get(status, (COLOR_TEXT_SECONDARY, COLOR_BG))
        chips.append(f"""
        <div style="position:absolute; top:{top_px:.0f}px; left:52px; right:6px; height:{chip_height_px:.0f}px;
                    background:{status_soft}; border-left:3px solid {status_color}; border-radius:8px;
                    padding:5px 10px; box-sizing:border-box; overflow:hidden;">
            <div style="font-family:{FONT_MONO}; font-size:10px; font-weight:700; color:{status_color};">{entry.get('Time', '—')}</div>
            <div style="font-size:12.5px; font-weight:700; color:{COLOR_TEXT_PRIMARY}; white-space:nowrap;
                        overflow:hidden; text-overflow:ellipsis;">{entry.get('Patient_ID', '—')}</div>
        </div>""")
    return f"""
    <div style="position:relative; height:{height_px}px; margin-left:44px; padding-bottom:4px;
                border-left:1px solid {COLOR_BORDER};">
        {''.join(rails)}
        {''.join(chips)}
    </div>"""
def format_agenda_row_html(entry):
    """The interactive counterpart to the calendar strip above — same time/
    status color coding, rendered as an ordinary card next to a real
    gr.Button (see app.py) so '1-click entry into that patient's session'
    stays a genuine Gradio click handler rather than an HTML onclick."""
    status = entry.get("Status", "Scheduled")
    status_color, status_soft = STATUS_COLORS.get(status, (COLOR_TEXT_SECONDARY, COLOR_BG))
    return f"""
    <div style="display:flex; align-items:center; justify-content:space-between; gap:16px;
                background:{COLOR_CARD}; border:1px solid {COLOR_BORDER}; border-radius:14px;
                padding:13px 18px; box-shadow:0 3px 10px rgba(16,34,59,0.04);">
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
HISTORY_TABLE_COLUMNS = ["Session #", "Date/Timestamp", "Risk Score", "Risk Level", "Dynamic Style", "Progress", "CPT Code", "Crisis Flag"]
def _format_session_date(record):
    """Only sessions logged live THIS browser session carry a real
    Session_Start (recsys.analyze_new_session() derives it from 'now') — the
    archived dataset has no real per-session date column (see Part 1 schema
    notes at the top of this module), so historical rows honestly show that
    rather than a fabricated date."""
    start = record.get("Session_Start")
    if start is not None:
        try:
            return start.strftime("%b %d, %Y · %H:%M")
        except AttributeError:
            pass
    return "— (archived, no timestamp)"
def format_patient_history_df(history_records):
    if not history_records:
        return pd.DataFrame(columns=HISTORY_TABLE_COLUMNS)
    rows = []
    for r in history_records:
        rows.append({
            "Session #": r.get("Session_Number"),
            "Date/Timestamp": _format_session_date(r),
            "Risk Score": r.get("Risk_Score", r.get("Predicted_Risk_Score", "—")),
            "Risk Level": r.get("Risk_Level", "—"),
            "Dynamic Style": r.get("Dynamic", r.get("Predicted_Dynamic", "—")),
            "Progress": r.get("Progress", "—"),
            "CPT Code": r.get("Target_CPT_Code", "—"),
            "Crisis Flag": "Yes" if r.get("Crisis_Flag") else "No",
        })
    return pd.DataFrame(rows)
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
# Ombré (white -> soft teal) plot background.
# Per design choice, the two clinical charts drop the former multi-color /
# monochrome-band backgrounds in favor of a single LIGHT gradient fill in the
# plot area — white at the top fading to a soft teal glow at the bottom, in
# the logo's tones. Because the plot area is light, every foreground element
# stays dark/teal (see _style_axes) so it reads clearly; the surrounding
# white card blends seamlessly into the figure margins.
# ==============================================================================
OMBRE_TOP = "#ffffff"        # white (top of the plot)
OMBRE_BOTTOM = "#cdeeed"     # soft teal glow (bottom)
_OMBRE_CMAP = LinearSegmentedColormap.from_list("navio_ombre", [OMBRE_BOTTOM, OMBRE_TOP])
def _draw_ombre(ax, xlim, ylim):
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect="auto",
              cmap=_OMBRE_CMAP, origin="lower", zorder=0, interpolation="bilinear")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_autoscale_on(False)  # keep later plot() calls from rescaling the ombré
# ==============================================================================
# GRAPH 1 — Cross-Session Progress & Trajectory. X: Session Number, Y: 0-10
# Risk_Score. A clean longitudinal line only — no crisis "X" clutter here by
# design (that belongs to Graph 2's turn-level detail); the ONLY distinct
# marker on this graph is a star, either for a live in-progress new session
# or for a specific historical session the clinician selected in the
# Patients page (mutually exclusive uses of the same visual language).
# ==============================================================================
def build_risk_trajectory_figure(history_records, new_session_result=None, highlight_session_number=None):
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=120)
    if not history_records and new_session_result is None:
        return _empty_figure("No trajectory yet — this will be the baseline session")
    x_hist = [r.get("Session_Number") for r in history_records]
    y_hist = [r.get("Risk_Score") for r in history_records]
    xs_all = [x for x in x_hist if x is not None]
    new_x = new_y = None
    if new_session_result is not None:
        new_x = next_session_number(history_records)
        new_y = new_session_result["Predicted_Risk_Score"]
        xs_all.append(new_x)
    if not xs_all:
        xs_all = [1]
    xmin, xmax = min(xs_all), max(xs_all)
    pad = max(0.5, (xmax - xmin) * 0.06)
    # Ombré (black -> teal) plot background, per design choice — replaces the
    # former risk-zone bands. The plot area is a teal-dominant gradient; the
    # white card shows through only in the figure margins.
    _draw_ombre(ax, (xmin - pad, xmax + pad), (-0.3, 10.3))
    if x_hist:
        ax.plot(x_hist, y_hist, color=COLOR_ACCENT, linewidth=2.2, marker="o",
                markersize=5.5, markerfacecolor="white", markeredgecolor=COLOR_ACCENT,
                markeredgewidth=1.8, zorder=3, label="Recorded sessions")
    # Selected / current session marker — a small filled teal dot with a white
    # halo ring (replaces the former star), for either a live new session or a
    # historical session selected in the Patients page.
    selected = None
    if new_session_result is not None:
        if x_hist:
            ax.plot([x_hist[-1], new_x], [y_hist[-1], new_y], color=COLOR_TEXT_MUTED,
                     linewidth=1.4, linestyle="--", zorder=2)
        selected = (new_x, new_y, "Current session")
    elif highlight_session_number is not None and highlight_session_number in x_hist:
        idx = x_hist.index(highlight_session_number)
        selected = (x_hist[idx], y_hist[idx], "Selected session")
    if selected is not None:
        ax.scatter([selected[0]], [selected[1]], s=150, color="#ffffff",
                   edgecolors=COLOR_ACCENT, linewidths=1.4, zorder=4)  # white halo w/ teal rim
        ax.scatter([selected[0]], [selected[1]], s=54, color=COLOR_ACCENT,
                   edgecolors="#ffffff", linewidths=1.0, zorder=5, label=selected[2])  # small teal core
    ax.set_xlabel("Session #", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("Risk Score", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    _style_axes(fig, ax)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=COLOR_TEXT_SECONDARY)
    fig.tight_layout()
    return fig
# ==============================================================================
# GRAPH 2 — Intra-Session Clinical Timeline. X: minutes 0 -> session duration
# (ALWAYS the full axis — see _rescale_turns below). Plots a per-turn
# "intensity" line with Safe/Alert/Flooding zone bands, colors each turn
# marker by speaker role, and adds clinical annotations grounded directly in
# the transcript evidence: crisis keywords (with the matched term(s)),
# abrupt sentiment shifts, and extended silences between turns.
# ==============================================================================
def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
def _rescale_turns(timeline_turns, duration_minutes):
    """The transcript's literal [MM:SS] timestamps often only span a short
    excerpt of the full session (true of every built-in Quick Starter, and
    common for partial real transcripts too) — plotted literally, every
    turn lands in the first minute or two and the rest of a 45/60-minute
    axis sits empty. This stretches parsed turn times proportionally across
    [0, duration_minutes] so the curve always fills the axis, while keeping
    each turn's ORIGINAL mm:ss (unscaled) for annotation text, so labels
    still show the real transcript timestamp, not the rescaled position."""
    raw_times = [t["time_min"] for t in timeline_turns]
    raw_span = max(raw_times) if raw_times else 0.0
    scale = (duration_minutes / raw_span) if raw_span > 0 else 1.0
    scaled = []
    for t in timeline_turns:
        scaled.append({**t, "plot_x": t["time_min"] * scale})
    return scaled, raw_span, scale
def build_intrasession_timeline_figure(timeline_turns, duration_minutes, crisis_flag=False):
    if not timeline_turns:
        return _empty_figure("No [MM:SS] timestamps found in this transcript —\nGraph 2 requires timestamped turns, e.g. \"[00:12] Dr. Carter: ...\"")
    duration_minutes = float(duration_minutes or 0) or 45.0
    turns, raw_span, scale = _rescale_turns(timeline_turns, duration_minutes)
    fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=120)
    # Ombré (black -> teal) plot background, per design choice — replaces the
    # former Safe/Alert/Flooding color bands. Those zones stay legible via the
    # light text labels below, drawn over the gradient.
    _draw_ombre(ax, (-0.5, duration_minutes + 0.5), (-4, 108))
    for y, txt in [(20, "Safe"), (55, "Alert"), (85, "Flooding")]:
        ax.text(duration_minutes, y, txt, fontsize=7, color=COLOR_TEXT_MUTED, ha="right", va="center",
                 style="italic", zorder=2)
    xs = [t["plot_x"] for t in turns]
    ys = [t["intensity"] for t in turns]
    ax.plot(xs, ys, color=COLOR_SECONDARY, linewidth=1.8, zorder=3, alpha=0.9)
    for t in turns:
        is_crisis = bool(t["crisis_hits"])
        role_color = COLOR_ACCENT if t["role"] == "patient" else "#8a94a6"
        marker = "X" if is_crisis else "o"
        size = 130 if is_crisis else 46
        face = COLOR_DANGER if is_crisis else role_color
        ax.scatter([t["plot_x"]], [t["intensity"]], s=size, marker=marker, color=face,
                   edgecolors="white", linewidths=1.0, zorder=5 if is_crisis else 4)
    # --- Clinical annotations, grounded in the transcript evidence that
    # actually drives these values (not a separate guess): crisis keywords
    # (the exact matched terms), abrupt sentiment/intensity shifts between
    # consecutive turns, and extended silences (large real time-gaps).
    crisis_turns = [t for t in turns if t["crisis_hits"]]
    labeled_crisis = sorted(crisis_turns, key=lambda t: -t["intensity"])[:3]  # cap text labels to avoid overlap
    for t in crisis_turns:
        ax.axvline(t["plot_x"], color=COLOR_DANGER, linewidth=1.0, linestyle="--", alpha=0.55, zorder=2)
    for t in labeled_crisis:
        minute, second = divmod(int(round(t["time_min"] * 60)), 60)
        terms = ", ".join(t["crisis_hits"][:2])
        ax.annotate(f"{minute:02d}:{second:02d} · {terms}", xy=(t["plot_x"], t["intensity"]),
                    xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=7, fontweight="bold", color=COLOR_DANGER_TEXT)
    shift_threshold = 30
    for i in range(1, len(turns)):
        if turns[i]["crisis_hits"]:
            continue  # already annotated as a crisis turn — avoid double-labeling
        delta = turns[i]["intensity"] - turns[i - 1]["intensity"]
        if abs(delta) >= shift_threshold:
            marker = "^" if delta > 0 else "v"
            ax.scatter([turns[i]["plot_x"]], [turns[i]["intensity"]], marker=marker, s=70,
                       color=COLOR_WARNING, edgecolors="white", linewidths=0.8, zorder=4)
            if abs(delta) >= 45:
                direction = "Escalation" if delta > 0 else "De-escalation"
                ax.annotate(direction, xy=(turns[i]["plot_x"], turns[i]["intensity"]),
                            xytext=(0, -13), textcoords="offset points", ha="center",
                            fontsize=6.5, color=COLOR_WARNING_TEXT, fontweight="bold")
    raw_times = [t["time_min"] for t in turns]
    gaps = [raw_times[i + 1] - raw_times[i] for i in range(len(raw_times) - 1)]
    median_gap = _median(gaps)
    for i, gap in enumerate(gaps):
        if gap >= 1.0 and (median_gap == 0 or gap >= 2.5 * median_gap):
            x0, x1 = turns[i]["plot_x"], turns[i + 1]["plot_x"]
            ax.axvspan(x0, x1, color=COLOR_TEXT_MUTED, alpha=0.06, zorder=1)
            ax.text((x0 + x1) / 2, 4, f"Pause ~{gap:.0f}m", ha="center", va="bottom",
                    fontsize=6.5, color=COLOR_TEXT_MUTED, style="italic")
    if raw_span > 0 and duration_minutes > raw_span * 1.15:
        ax.text(0.0, 1.06, f"Turns scaled proportionally to fill the {duration_minutes:.0f}-minute session axis",
                transform=ax.transAxes, fontsize=7, color=COLOR_TEXT_MUTED, style="italic")
    legend_handles = [
        mlines.Line2D([], [], color=COLOR_ACCENT, marker="o", linestyle="None", markersize=6, label="Patient turn"),
        mlines.Line2D([], [], color="#8a94a6", marker="o", linestyle="None", markersize=6, label="Therapist turn"),
    ]
    if crisis_turns:
        legend_handles.append(
            mlines.Line2D([], [], color=COLOR_DANGER, marker="X", linestyle="None", markersize=8, label="Crisis keyword")
        )
    ax.set_xlabel("Session Time (minutes)", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    ax.set_ylabel("Estimated Intensity", fontsize=9, color=COLOR_TEXT_SECONDARY, labelpad=8)
    _style_axes(fig, ax)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5, frameon=False,
              labelcolor=COLOR_TEXT_SECONDARY, ncol=len(legend_handles))
    fig.tight_layout()
    return fig
