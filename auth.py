"""
NavIO — auth.py
Demo clinician login + doctor-scoped patient/schedule assignment.

This simulates an EMR-style login STATE within the app; it is NOT production
access control. A public Hugging Face Space serves the same running process
to every visitor, so this gate only scopes what's shown in the UI during a
session — it does not secure PHI. Worth stating plainly in any write-up of
this project.

The credentials below are entirely fictional, generated for this demo only.
Deliberately not using real Teudat Zehut numbers (even as placeholders) —
there's no reason to put real national ID numbers anywhere near a health-data
demo, fictional or not.
"""

import hashlib

# --- Demo clinician roster (fictional) ---------------------------------------
DEMO_DOCTORS = [
    {"doctor_id": 0, "name": "Dr. Sarah Cohen", "id_number": "000000010"},
    {"doctor_id": 1, "name": "Dr. David Levi", "id_number": "000000028"},
    {"doctor_id": 2, "name": "Dr. Maya Ben-Ari", "id_number": "000000036"},
]

NUM_DOCTORS = len(DEMO_DOCTORS)


def _normalize(value):
    return str(value).strip().lower()


def authenticate(name, id_number):
    """Returns the matching doctor dict on success, or None on failure.
    Case/whitespace-tolerant match against the demo roster."""
    name_norm, id_norm = _normalize(name), _normalize(id_number)
    if not name_norm or not id_norm:
        return None
    for doctor in DEMO_DOCTORS:
        if _normalize(doctor["name"]) == name_norm and _normalize(doctor["id_number"]) == id_norm:
            return doctor
    return None


def _stable_bucket(patient_id, num_buckets):
    """Deterministic, process-independent hash — Python's built-in hash() is
    randomized per-process for strings (PYTHONHASHSEED), which would reassign
    every patient to a different doctor on every Space restart. This doesn't."""
    digest = hashlib.md5(str(patient_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % num_buckets


def get_assigned_patient_ids(doctor_id, all_patient_ids):
    """Every patient deterministically belongs to exactly one demo doctor,
    so 'scoping' is real rather than cosmetic (every doctor seeing the full
    2,200-patient roster would defeat the point)."""
    return [pid for pid in all_patient_ids if _stable_bucket(pid, NUM_DOCTORS) == doctor_id]


def generate_mock_schedule(doctor_id, assigned_patient_ids, max_slots=8):
    """NavIO's dataset models historical therapy sessions, not future
    appointment bookings — there's no real scheduling data to show on a
    Dashboard. This synthesizes a small, deterministic 'today' schedule purely
    for that view, clearly separate from the real historical archive shown
    inside a patient's profile."""
    slot_times = ["09:00", "09:45", "10:30", "11:15", "13:00", "13:45", "14:30", "15:15"]
    chosen = sorted(assigned_patient_ids)[:max_slots]
    schedule = []
    for i, patient_id in enumerate(chosen):
        status_bucket = _stable_bucket(f"{patient_id}-status", 3)
        status = ["Completed", "Scheduled", "Scheduled"][status_bucket]
        schedule.append({
            "Time": slot_times[i] if i < len(slot_times) else f"Slot {i + 1}",
            "Patient_ID": patient_id,
            "Status": status,
        })
    return schedule
