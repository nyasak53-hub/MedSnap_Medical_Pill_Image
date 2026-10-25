import os
import io
import re
import json
import time
import base64
import smtplib
import tempfile
import threading
import webbrowser
from collections import deque
from copy import deepcopy
from datetime import datetime, date, time as dtime, timedelta
from email.message import EmailMessage
from functools import wraps
from zoneinfo import ZoneInfo

import httpx
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageOps

# ==========================================================================
# Setup
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# static_folder=None: pages are served only through the explicit routes below,
# so app.py, .env and your data file can never be downloaded by URL.
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
CORS(app, resources={r"/api/*": {"origins": "*"}})

load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
DATA_FILE = os.path.join(DATA_DIR, "medications_data.json")
DATA_LOCK = threading.RLock()


def normalize_secret(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip().strip('"').strip("'")
    placeholders = {"your_groq_api_key_here", "your_real_key_here", "replace_me", "changeme", ""}
    return None if value.lower() in placeholders else value


GROQ_API_KEY = normalize_secret(os.environ.get("GROQ_API_KEY"))
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as exc:
        print(f"[warn] Groq client initialization failed: {exc}")
else:
    print("[warn] GROQ_API_KEY not set; AI features are disabled.")

# Change these from Render's Environment tab without touching code.
# Groq retires models from time to time (llama-3.3-70b-versatile and llama-4-scout were shut down in 2026),
# so each list has fallbacks. A model set in .env / Render is tried first. See console.groq.com/docs/models
def _model_list(env_name, defaults):
    custom = [m.strip() for m in os.environ.get(env_name, "").split(",") if m.strip()]
    return list(dict.fromkeys(custom + defaults))


TEXT_MODELS = _model_list("GROQ_TEXT_MODEL", ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
VISION_MODELS = _model_list("GROQ_VISION_MODELS", ["qwen/qwen3.8-27b"])
TEXT_MODEL = TEXT_MODELS[0]

# Caregiver alert delivery (all optional; features switch on when configured)
SMTP_HOST = normalize_secret(os.environ.get("SMTP_HOST"))
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = normalize_secret(os.environ.get("SMTP_USER"))
SMTP_PASS = normalize_secret(os.environ.get("SMTP_PASS"))
SMTP_FROM = normalize_secret(os.environ.get("SMTP_FROM")) or SMTP_USER
TWILIO_SID = normalize_secret(os.environ.get("TWILIO_ACCOUNT_SID"))
TWILIO_TOKEN = normalize_secret(os.environ.get("TWILIO_AUTH_TOKEN"))
TWILIO_FROM = normalize_secret(os.environ.get("TWILIO_FROM"))
MAX_ALERTS_PER_HOUR = int(os.environ.get("MAX_ALERTS_PER_HOUR", "30"))

EMAIL_CONFIGURED = bool(SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_FROM)
SMS_CONFIGURED = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM)

VALID_SLOTS = ("Morning", "Afternoon", "Evening")
SLOT_DEFAULT_TIME = {"Morning": "08:00", "Afternoon": "13:00", "Evening": "20:00"}
VALID_BADGES = {"with_food", "empty_stomach", "avoid_dairy", "avoid_alcohol", "avoid_sugar",
                "before_bed", "with_water", "before_lunch", "before_breakfast", "before_dinner",
                "avoid_grapefruit"}  # avoid_grapefruit kept so older saved medications still load
ALL_DAYS = [0, 1, 2, 3, 4, 5, 6]  # Monday = 0 ... Sunday = 6
# A dose counts as missed (and the caregiver is alerted) this many minutes after its time.
MISSED_GRACE_MINUTES = int(os.environ.get("MISSED_DOSE_GRACE_MINUTES", "30"))
VALID_SHAPES = {"", "round", "oval", "capsule", "oblong", "square", "diamond", "triangle", "hexagon"}
VALID_COLORS = {"", "white", "yellow", "orange", "pink", "red", "brown", "green", "blue", "purple", "gray"}
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Simplified Chinese"}
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

DEFAULT_MEDS = [
    {"id": 1, "name": "Metformin", "dosage": "500mg", "slot": "Morning", "stock": 14,
     "daily_dose": 2, "food_guide": "Take with your largest meal to avoid stomach upset.",
     "badges": ["with_food"], "scheduled_time": "08:00", "critical": False,
     "pill": {"shape": "oval", "color": "white", "imprint": "MET 500"}},
    {"id": 2, "name": "Lisinopril", "dosage": "10mg", "slot": "Evening", "stock": 4,
     "daily_dose": 1, "food_guide": "Can be taken with or without food.",
     "badges": [], "scheduled_time": "20:00", "critical": False,
     "pill": {"shape": "", "color": "", "imprint": ""}},
]

SAMPLE_PILL_DB = [
    {"name": "Aspirin 81mg", "shape": "round", "color": "white", "imprint": "BAYER"},
    {"name": "Lipitor 20mg", "shape": "oval", "color": "white", "imprint": "PD 155"},
    {"name": "Ibuprofen 200mg", "shape": "round", "color": "brown", "imprint": "I-2"},
    {"name": "Metformin 500mg", "shape": "oval", "color": "white", "imprint": "MET 500"},
]


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


# ==========================================================================
# Small helpers
# ==========================================================================
def clean_text(value, default="", max_len=120):
    text = str(value if value is not None else default).strip()
    return (text or default)[:max_len]


def clamp_int(value, default, low, high):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def normalize_slot(value):
    text = str(value or "").strip().lower()
    for slot in VALID_SLOTS:
        if slot.lower() in text:
            return slot
    return "Morning"


def valid_tz(name):
    name = str(name or "").strip()
    if not name or len(name) > 64:
        return None
    try:
        ZoneInfo(name)
        return name
    except Exception:
        return None


def request_tz():
    return valid_tz(request.headers.get("X-Timezone")) or "UTC"


def client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.remote_addr or "unknown"


def clean_pill(raw):
    raw = raw if isinstance(raw, dict) else {}
    shape = str(raw.get("shape", "")).lower()
    color = str(raw.get("color", "")).lower()
    return {
        "shape": shape if shape in VALID_SHAPES else "",
        "color": color if color in VALID_COLORS else "",
        "imprint": clean_text(raw.get("imprint"), "", 12).upper(),
    }


# ==========================================================================
# Storage (one state object per browser, keyed by X-Client-Id)
# ==========================================================================
def new_state():
    return {
        "meds": deepcopy(DEFAULT_MEDS),
        "history": {},
        "caregiver": {"enabled": False, "consent": False, "patient_name": "",
                      "name": "", "email": "", "phone": ""},
        "tz": "UTC",
        "best_streak": 0,
        "profile": {"name": "", "age": "", "language": "en", "allergies": "",
                    "timezone": "", "low_stock_days": 7},
        "low_alerted": [],
        "low_tries": {},
    }


def migrate_state(raw):
    """Accepts None, an older list-of-meds format, or the current dict format."""
    if raw is None:
        return new_state()
    if isinstance(raw, list):
        state = new_state()
        state["meds"] = [m for m in raw if isinstance(m, dict) and "id" in m]
    elif isinstance(raw, dict):
        state = raw
    else:
        return new_state()
    base = new_state()
    for key in ("history", "caregiver", "tz", "best_streak", "profile", "low_alerted", "low_tries"):
        state.setdefault(key, base[key])
    for key, value in base["profile"].items():
        state["profile"].setdefault(key, value)
    state.setdefault("meds", [])
    for med in state["meds"]:
        med.pop("taken", None)
        med.setdefault("badges", [])
        med.setdefault("critical", False)
        med.setdefault("pill", {"shape": "", "color": "", "imprint": ""})
        med.setdefault("scheduled_time", SLOT_DEFAULT_TIME.get(med.get("slot"), "09:00"))
        med.setdefault("food_guide", "Take as directed by your physician.")
        med.setdefault("daily_dose", 1)
        med.setdefault("stock", 0)
        med.setdefault("days", list(ALL_DAYS))
        med.setdefault("instructions", "")
        med.setdefault("created", date.today().isoformat())
    return state


def _read_store():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            print(f"[warn] Could not read data file: {exc}")
    return {}


def _write_store(store):
    os.makedirs(DATA_DIR, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(store, f)
        os.replace(tmp_path, DATA_FILE)  # atomic: never leaves a half-written file
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def mutate(client_id, fn):
    """Load this client's state, run fn(state), save only if it changed."""
    with DATA_LOCK:
        store = _read_store()
        before = json.dumps(store.get(client_id), sort_keys=True)
        state = migrate_state(store.get(client_id))
        result = fn(state)
        if client_id not in store or json.dumps(state, sort_keys=True) != before:
            store[client_id] = state
            try:
                _write_store(store)
            except Exception as exc:
                print(f"[warn] Could not save data: {exc}")
        return result


def require_client(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        cid = request.headers.get("X-Client-Id", "")
        if not CLIENT_ID_RE.match(cid):
            return jsonify({"success": False, "error": "Missing or invalid client id."}), 400
        return fn(cid, *args, **kwargs)
    return wrapper


# ==========================================================================
# Daily records, adherence streak
# ==========================================================================
def scheduled_on(med, day):
    """True if this medication is taken on that date (its weekday is ticked and it existed then)."""
    return day.weekday() in (med.get("days") or ALL_DAYS) and day.isoformat() >= str(med.get("created", ""))


def get_record(state, day):
    hist = state["history"]
    key = day.isoformat()
    rec = hist.get(key)
    if rec is None:
        rec = hist[key] = {"due": [m["id"] for m in state["meds"] if scheduled_on(m, day)],
                           "taken": [], "alerted": [], "tries": {}, "used": {}}
    for field, default in (("due", []), ("taken", []), ("alerted", []), ("tries", {}), ("used", {})):
        rec.setdefault(field, deepcopy(default))
    return rec


def ensure_today(state, today):
    hist = state["history"]
    rec = get_record(state, today)
    for med in state["meds"]:
        if med["id"] not in rec["due"] and scheduled_on(med, today):
            rec["due"].append(med["id"])
    if len(hist) > 400:  # keep roughly the last year
        for old in sorted(hist.keys())[:-400]:
            del hist[old]
    return rec


def day_info(state, day, today):
    """(due ids, taken ids) for any date. Past days without a record use the schedule."""
    ids = {m["id"] for m in state["meds"]}
    rec = state["history"].get(day.isoformat())
    if rec and day <= today:
        due = [i for i in rec.get("due", []) if i in ids]
        taken = [i for i in rec.get("taken", []) if i in due]
    else:
        due = [m["id"] for m in state["meds"] if scheduled_on(m, day)]
        taken = []
    return due, taken


def earliest_day(state, today):
    days = []
    for m in state["meds"]:
        try:
            days.append(date.fromisoformat(str(m.get("created"))))
        except ValueError:
            pass
    return max(min(days, default=today), today - timedelta(days=400))


def compute_streak(state, today):
    """Consecutive days where every scheduled dose was taken. Days with nothing scheduled are skipped."""
    streak, day, first = 0, today, earliest_day(state, today)
    while day >= first:
        due, taken = day_info(state, day, today)
        if due:
            if set(due) <= set(taken):
                streak += 1
            elif day != today:  # today still in progress doesn't break the streak
                break
        day -= timedelta(days=1)
    return streak


def avg_daily_use(med):
    return max(1, int(med.get("daily_dose", 1))) * len(med.get("days") or ALL_DAYS) / 7


def days_left(med):
    return med["stock"] / avg_daily_use(med)


def low_stock_days(state):
    return clamp_int(state["profile"].get("low_stock_days"), 7, 1, 30)


def is_low(state, med):
    return days_left(med) < low_stock_days(state)


def week_view(state, start, today):
    meds = {m["id"]: m for m in state["meds"]}
    out = []
    for i in range(7):
        day = start + timedelta(days=i)
        due, taken = day_info(state, day, today)
        items = sorted(({"id": mid, "name": meds[mid]["name"], "dosage": meds[mid]["dosage"],
                         "time": meds[mid]["scheduled_time"], "pills": meds[mid]["daily_dose"],
                         "taken": mid in taken, "pill": meds[mid]["pill"]} for mid in due),
                       key=lambda x: x["time"])
        if not due:
            status = "none"
        elif day > today:
            status = "planned"
        elif set(due) <= set(taken):
            status = "complete"
        elif day == today:
            status = "today"
        else:
            status = "missed"
        out.append({"date": day.isoformat(), "items": items, "taken": len(taken),
                    "total": len(due), "status": status})
    return out


def build_view(state, today, extra=None):
    rec = ensure_today(state, today)
    taken = set(rec["taken"])
    alerted = set(rec["alerted"])
    due_today = set(rec["due"])
    meds = [dict(m, taken=m["id"] in taken, alerted=m["id"] in alerted, due_today=m["id"] in due_today,
                 days_left=round(days_left(m), 1), low=is_low(state, m)) for m in state["meds"]]
    streak = compute_streak(state, today)
    state["best_streak"] = max(state.get("best_streak", 0), streak)
    view = {
        "success": True,
        "medications": meds,
        "streak": streak,
        "best_streak": state["best_streak"],
        "caregiver": state["caregiver"],
        "alerts_available": {"email": EMAIL_CONFIGURED, "sms": SMS_CONFIGURED},
        "ai_enabled": client is not None,
        "profile": state["profile"],
        "missed_grace_minutes": MISSED_GRACE_MINUTES,
        "today": today.isoformat(),
        "week": week_view(state, today - timedelta(days=today.weekday()), today),
    }
    if extra:
        view.update(extra)
    return view


def api_op(client_id, op):
    """Run op(state, today) inside a saved transaction and return the fresh view."""
    def wrapped(state):
        # The time zone saved in Profile wins; otherwise use the browser's.
        if not valid_tz(state["profile"].get("timezone")):
            state["profile"]["timezone"] = request_tz()
        tzname = valid_tz(state["profile"]["timezone"]) or "UTC"
        today = datetime.now(ZoneInfo(tzname)).date()
        state["tz"] = tzname
        ensure_today(state, today)
        extra = op(state, today)
        return build_view(state, today, extra)

    return jsonify(mutate(client_id, wrapped))


def find_med(state, med_id):
    for med in state["meds"]:
        if med["id"] == med_id:
            return med
    raise ApiError("Medication not found", 404)


# ==========================================================================
# Pages
# ==========================================================================
def page_response(filename):
    if os.path.isfile(os.path.join(BASE_DIR, filename)):
        return send_from_directory(BASE_DIR, filename)
    return None


@app.route("/")
def landing():
    return (
        page_response("landing.html")
        or page_response("index.html")
        or Response(
            "<h1>MedSnap is running</h1><p>The server started, but <code>index.html</code> was not "
            "found next to <code>app.py</code>. Make sure it is committed and pushed to GitHub.</p>",
            status=500, mimetype="text/html")
    )


@app.route("/app")
def main_app():
    return page_response("index.html") or landing()


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "index_html_found": os.path.isfile(os.path.join(BASE_DIR, "index.html")),
        "landing_html_found": os.path.isfile(os.path.join(BASE_DIR, "landing.html")),
        "ai_enabled": client is not None,
        "email_alerts": EMAIL_CONFIGURED,
        "sms_alerts": SMS_CONFIGURED,
    })


# ==========================================================================
# Medication API
# ==========================================================================
@app.route("/api/state", methods=["GET"])
@require_client
def get_state(cid):
    return api_op(cid, lambda state, today: None)


def parse_medication(data):
    """Validate the Add/Edit medication form."""
    name = clean_text(data.get("name"))
    if not name:
        raise ApiError("Medication name is required.")
    sched = str(data.get("scheduled_time") or "")
    if not TIME_RE.match(sched):
        raise ApiError("Please choose a dose time.")
    hour = int(sched[:2])
    slot = "Morning" if hour < 12 else "Afternoon" if hour < 17 else "Evening"
    days = sorted({int(d) for d in (data.get("days") or []) if str(d).isdigit() and 0 <= int(d) <= 6})
    if not days:
        raise ApiError("Please tick at least one day of the week.")
    badges = [b for b in (data.get("badges") or []) if b in VALID_BADGES]
    return {
        "name": name,
        "dosage": clean_text(data.get("dosage"), "", 60),
        "slot": slot,
        "stock": clamp_int(data.get("stock"), 30, 0, 10000),
        "daily_dose": clamp_int(data.get("daily_dose"), 1, 1, 20),
        "instructions": clean_text(data.get("instructions"), "", 300),
        "food_guide": clean_text(data.get("food_guide"), "", 200),
        "badges": list(dict.fromkeys(badges)),
        "days": days,
        "scheduled_time": sched,
        "critical": bool(data.get("critical")),
        "pill": clean_pill(data.get("pill")),
    }


@app.route("/api/week", methods=["GET"])
@require_client
def get_week(cid):
    """Calendar: any week (start = the Monday, YYYY-MM-DD)."""
    try:
        start = date.fromisoformat(request.args.get("start", ""))
    except ValueError:
        raise ApiError("Use a date like 2026-10-19.")

    def op(state, today):
        monday = start - timedelta(days=start.weekday())
        return {"week_start": monday.isoformat(), "week_days": week_view(state, monday, today)}

    return api_op(cid, op)


@app.route("/api/medications", methods=["POST"])
@require_client
def add_medication(cid):
    data = request.get_json(silent=True) or {}

    def op(state, today):
        if len(state["meds"]) >= 100:
            raise ApiError("Medication limit reached.")
        med = parse_medication(data)
        med["id"] = max([m["id"] for m in state["meds"]], default=0) + 1
        med["created"] = today.isoformat()
        state["meds"].append(med)
        ensure_today(state, today)
        return None

    return api_op(cid, op)


@app.route("/api/medications/<int:med_id>", methods=["PUT"])
@require_client
def edit_medication(cid, med_id):
    data = request.get_json(silent=True) or {}

    def op(state, today):
        med = find_med(state, med_id)
        med.update(parse_medication(data))
        rec = state["history"][today.isoformat()]
        if scheduled_on(med, today):
            if med_id not in rec["due"]:
                rec["due"].append(med_id)
        elif med_id not in rec["taken"]:
            rec["due"] = [i for i in rec["due"] if i != med_id]
        return None

    return api_op(cid, op)


@app.route("/api/medications/<int:med_id>", methods=["DELETE"])
@require_client
def delete_medication(cid, med_id):
    def op(state, today):
        med = find_med(state, med_id)
        state["meds"].remove(med)
        rec = state["history"][today.isoformat()]
        rec["due"] = [i for i in rec["due"] if i != med_id]
        rec["taken"] = [i for i in rec["taken"] if i != med_id]
        return None

    return api_op(cid, op)


@app.route("/api/medications/<int:med_id>/take", methods=["POST"])
@require_client
def take_medication(cid, med_id):
    data = request.get_json(silent=True) or {}

    def op(state, today):
        med = find_med(state, med_id)
        day = today
        if data.get("date"):
            try:
                day = date.fromisoformat(str(data["date"]))
            except ValueError:
                raise ApiError("Invalid date.")
        if day > today:
            raise ApiError("You can't mark a future dose as taken.")
        if (today - day).days > 60:
            raise ApiError("That day is too far back to change.")
        rec = get_record(state, day)
        if med_id not in rec["due"]:
            if not scheduled_on(med, day):
                raise ApiError("This medication isn't scheduled on that day.")
            rec["due"].append(med_id)
        if med_id in rec["taken"]:  # undo: give back exactly what was removed
            rec["taken"].remove(med_id)
            med["stock"] += int(rec["used"].pop(str(med_id), 0))
        else:
            rec["taken"].append(med_id)
            used = min(med["stock"], med["daily_dose"])
            med["stock"] -= used
            rec["used"][str(med_id)] = used
        return None

    return api_op(cid, op)


@app.route("/api/medications/<int:med_id>/refill", methods=["POST"])
@require_client
def refill_medication(cid, med_id):
    data = request.get_json(silent=True) or {}

    def op(state, today):
        med = find_med(state, med_id)
        if data.get("set") is not None and str(data.get("set")).strip() != "":
            med["stock"] = clamp_int(data.get("set"), med["stock"], 0, 10000)
            return None
        amount = clamp_int(data.get("amount"), 0, 0, 10000)
        if amount <= 0:
            raise ApiError("Enter how many pills you added.")
        med["stock"] = min(10000, med["stock"] + amount)
        return None

    return api_op(cid, op)


@app.route("/api/profile", methods=["PUT"])
@require_client
def save_profile(cid):
    data = request.get_json(silent=True) or {}

    def op(state, today):
        age = str(data.get("age") or "").strip()
        if age and not (age.isdigit() and 1 <= int(age) <= 120):
            raise ApiError("Age should be a number between 1 and 120.")
        tz = str(data.get("timezone") or "").strip()
        if tz and not valid_tz(tz):
            raise ApiError("That time zone isn't recognised. Try something like America/New_York.")
        lang = data.get("language") if data.get("language") in LANGUAGE_NAMES else "en"
        state["profile"] = {
            "name": clean_text(data.get("name"), "", 60),
            "age": age,
            "language": lang,
            "allergies": clean_text(data.get("allergies"), "", 500),
            "timezone": tz or state["profile"].get("timezone") or "UTC",
            "low_stock_days": clamp_int(data.get("low_stock_days"), 7, 1, 30),
        }
        if state["profile"]["name"]:
            state["caregiver"]["patient_name"] = state["profile"]["name"]
        return None

    return api_op(cid, op)


# ==========================================================================
# Caregiver alerts
# ==========================================================================
SEND_LOG = deque()
IP_TEST_LOG = {}
RATE_LOCK = threading.Lock()


def global_send_allowed():
    now = time.time()
    with RATE_LOCK:
        while SEND_LOG and now - SEND_LOG[0] > 3600:
            SEND_LOG.popleft()
        if len(SEND_LOG) >= MAX_ALERTS_PER_HOUR:
            return False
        SEND_LOG.append(now)
        return True


def ip_test_allowed(ip, limit=3):
    now = time.time()
    with RATE_LOCK:
        stamps = [t for t in IP_TEST_LOG.get(ip, []) if now - t < 3600]
        if len(stamps) >= limit:
            IP_TEST_LOG[ip] = stamps
            return False
        stamps.append(now)
        IP_TEST_LOG[ip] = stamps
        return True


def send_email(to_addr, subject, body):
    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)


def send_sms(to_number, body):
    response = httpx.post(
        f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
        data={"To": to_number, "From": TWILIO_FROM, "Body": body},
        auth=(TWILIO_SID, TWILIO_TOKEN),
        timeout=15,
    )
    response.raise_for_status()


def deliver_alert(caregiver, subject, body):
    """Try every configured channel. Returns a list of channels that succeeded."""
    delivered = []
    if EMAIL_CONFIGURED and caregiver.get("email"):
        try:
            send_email(caregiver["email"], subject, body)
            delivered.append("email")
        except Exception as exc:
            print(f"[error] alert email failed: {exc}")
    if SMS_CONFIGURED and caregiver.get("phone"):
        try:
            send_sms(caregiver["phone"], body[:300])
            delivered.append("sms")
        except Exception as exc:
            print(f"[error] alert SMS failed: {exc}")
    return delivered


@app.route("/api/caregiver", methods=["PUT"])
@require_client
def save_caregiver(cid):
    data = request.get_json(silent=True) or {}

    def op(state, today):
        enabled = bool(data.get("enabled"))
        email = clean_text(data.get("email"), "", 254)
        phone = re.sub(r"[\s()-]", "", clean_text(data.get("phone"), "", 30))
        if email and not EMAIL_RE.match(email):
            raise ApiError("That email address doesn't look right.")
        if phone and not PHONE_RE.match(phone):
            raise ApiError("Phone must be in international format, e.g. +14155550123.")
        if enabled:
            if not (email or phone):
                raise ApiError("Add a caregiver email or phone number.")
            if not data.get("consent"):
                raise ApiError("Please confirm your caregiver agreed to receive alerts.")
        before = state["caregiver"]
        state["caregiver"] = {
            "enabled": enabled,
            "consent": bool(data.get("consent")),
            "patient_name": clean_text(data.get("patient_name"), "", 60) or state["profile"].get("name", ""),
            "name": clean_text(data.get("name"), "", 60),
            "email": email,
            "phone": phone,
        }
        cg = state["caregiver"]
        newly_connected = enabled and (not before.get("enabled") or before.get("email") != email
                                       or before.get("phone") != phone)
        if newly_connected and (EMAIL_CONFIGURED or SMS_CONFIGURED):
            # Let the caregiver know right away that they'll get alerts (sent in the background).
            threading.Thread(target=send_welcome, args=(dict(cg),), daemon=True).start()
            return {"delivery": "sending"}
        return {"delivery": "connected" if (EMAIL_CONFIGURED or SMS_CONFIGURED) else "not_configured"}

    return api_op(cid, op)


def send_welcome(caregiver):
    if not global_send_allowed():
        return
    who = caregiver.get("patient_name") or "A MedSnap user"
    greet = f"Hi {caregiver['name']}, " if caregiver.get("name") else "Hi, "
    body = (f"{greet}{who} added you as their caregiver in MedSnap. You'll get a message right away "
            f"if they miss a dose or their pills are running low. No action is needed now.")
    deliver_alert(caregiver, "You're now a MedSnap caregiver", body)


@app.route("/api/caregiver/emergency", methods=["POST"])
@require_client
def emergency_caregiver(cid):
    if not (EMAIL_CONFIGURED or SMS_CONFIGURED):
        return jsonify({"success": False, "error": "Email/SMS sending isn't configured on this server yet."}), 503
    caregiver = mutate(cid, lambda state: dict(state["caregiver"], patient_name=state["caregiver"].get("patient_name") or state["profile"].get("name")))
    if not (caregiver.get("enabled") and caregiver.get("consent") and (caregiver.get("email") or caregiver.get("phone"))):
        return jsonify({"success": False, "error": "Turn on caregiver alerts and save their contact first."}), 400
    if not global_send_allowed():
        return jsonify({"success": False, "error": "Server message limit reached. Call them directly."}), 429
    who = caregiver.get("patient_name") or "A MedSnap user"
    body = (f"EMERGENCY: {who} pressed the Emergency button in MedSnap. Please contact them right away. "
            f"If you can't reach them, call 911.")
    delivered = deliver_alert(caregiver, f"EMERGENCY: {who} needs help", body)
    if not delivered:
        return jsonify({"success": False, "error": "The message could not be sent. Check the server logs."}), 502
    return jsonify({"success": True, "delivered": delivered})


@app.route("/api/caregiver/test", methods=["POST"])
@require_client
def test_caregiver(cid):
    if not (EMAIL_CONFIGURED or SMS_CONFIGURED):
        return jsonify({"success": False, "error": "Email/SMS sending isn't configured on this server yet."}), 503
    if not ip_test_allowed(client_ip()):
        return jsonify({"success": False, "error": "Too many test messages. Try again in an hour."}), 429

    caregiver = mutate(cid, lambda state: dict(state["caregiver"]))
    if not (caregiver.get("consent") and (caregiver.get("email") or caregiver.get("phone"))):
        return jsonify({"success": False, "error": "Save your caregiver details (with consent) first."}), 400
    if not global_send_allowed():
        return jsonify({"success": False, "error": "Server message limit reached. Try again later."}), 429

    who = caregiver.get("patient_name") or "A MedSnap user"
    body = (f"MedSnap test message: {who} listed you as a caregiver and will send you a note "
            f"if a critical medication isn't marked as taken. No action is needed.")
    delivered = deliver_alert(caregiver, "MedSnap test message", body)
    if not delivered:
        return jsonify({"success": False, "error": "The message could not be sent. Check the server logs."}), 502
    return jsonify({"success": True, "delivered": delivered})


ALERT_WINDOW_HOURS = 3  # still alert if the server only wakes up within 3h after the 60-minute mark


def check_missed_doses():
    """Every minute: alert the caregiver about missed doses and low stock (each only once)."""
    pending = []
    with DATA_LOCK:
        store = _read_store()
        changed = False
        for cid, raw in store.items():
            state = migrate_state(raw)
            cg = state["caregiver"]
            if not (cg.get("enabled") and cg.get("consent") and (cg.get("email") or cg.get("phone"))):
                continue
            tzname = valid_tz(state["profile"].get("timezone")) or valid_tz(state.get("tz")) or "UTC"
            now = datetime.now(ZoneInfo(tzname))
            rec = ensure_today(state, now.date())
            who = cg.get("patient_name") or state["profile"].get("name") or "A MedSnap user"
            greet = f"Hi {cg['name']}, " if cg.get("name") else ""

            for med in state["meds"]:
                if med["id"] not in rec["due"] or med["id"] in rec["taken"] or med["id"] in rec["alerted"]:
                    continue
                if int(rec["tries"].get(str(med["id"]), 0)) >= 3:
                    continue
                hh, mm = (int(x) for x in med["scheduled_time"].split(":"))
                due_at = datetime.combine(now.date(), dtime(hh, mm), tzinfo=ZoneInfo(tzname))
                deadline = due_at + timedelta(minutes=MISSED_GRACE_MINUTES)
                if deadline <= now < deadline + timedelta(hours=ALERT_WINDOW_HOURS):
                    important = " This is marked as an important medication." if med.get("critical") else ""
                    body = (f"{greet}{who} has not marked {med['name']} ({med['dosage'] or str(med['daily_dose']) + ' pill(s)'}) "
                            f"as taken. It was scheduled for {med['scheduled_time']}.{important} Please check in with them. "
                            f"(Sent by MedSnap because you were listed as a caregiver.)")
                    pending.append(("dose", cid, med["id"], now.date().isoformat(), dict(cg),
                                    f"MedSnap: missed dose of {med['name']}", body))

            for med in state["meds"]:
                key = str(med["id"])
                if is_low(state, med):
                    if med["id"] in state["low_alerted"] or int(state["low_tries"].get(key, 0)) >= 3:
                        continue
                    left = days_left(med)
                    body = (f"{greet}{who} is running low on {med['name']}: {med['stock']} pill(s) left, "
                            f"about {left:.0f} day(s) of doses. Please help them arrange a refill. "
                            f"(Sent by MedSnap because you were listed as a caregiver.)")
                    pending.append(("low", cid, med["id"], None, dict(cg),
                                    f"MedSnap: {med['name']} is running low", body))
                else:  # refilled: allow a new alert next time it runs low
                    if med["id"] in state["low_alerted"]:
                        state["low_alerted"].remove(med["id"])
                    state["low_tries"].pop(key, None)
            store[cid] = state
            changed = True
        if changed:
            try:
                _write_store(store)
            except Exception as exc:
                print(f"[warn] Could not save data: {exc}")

    for kind, cid, med_id, day_key, caregiver, subject, body in pending:
        if not (EMAIL_CONFIGURED or SMS_CONFIGURED) or not global_send_allowed():
            continue
        delivered = deliver_alert(caregiver, subject, body)

        def record(state, _kind=kind, _id=med_id, _day=day_key, _ok=bool(delivered)):
            if _kind == "low":
                if _ok:
                    if _id not in state["low_alerted"]:
                        state["low_alerted"].append(_id)
                else:
                    state["low_tries"][str(_id)] = int(state["low_tries"].get(str(_id), 0)) + 1
                return
            rec = state["history"].get(_day)
            if rec is None:
                return
            if _ok:
                if _id not in rec["alerted"]:
                    rec["alerted"].append(_id)
            else:
                rec["tries"][str(_id)] = int(rec["tries"].get(str(_id), 0)) + 1

        mutate(cid, record)


def alert_worker():
    while True:
        try:
            check_missed_doses()
        except Exception as exc:
            print(f"[error] alert worker: {exc}")
        time.sleep(60)


_worker_started = False


def start_alert_worker():
    global _worker_started
    if _worker_started or os.environ.get("ENABLE_ALERT_WORKER", "1") != "1":
        return
    _worker_started = True
    threading.Thread(target=alert_worker, name="alert-worker", daemon=True).start()


start_alert_worker()


# ==========================================================================
# AI: pill search, interactions, chat, translation, label scan
# ==========================================================================
@app.route("/api/identify-pill", methods=["POST"])
def identify_pill():
    data = request.get_json(silent=True) or {}
    shape = str(data.get("shape", "")).lower().strip()
    color = str(data.get("color", "")).lower().strip()
    imprint = re.sub(r"[\s-]", "", str(data.get("imprint", "")).upper())
    matches = [
        p["name"] for p in SAMPLE_PILL_DB
        if (not shape or shape == p["shape"])
        and (not color or color == p["color"])
        and (not imprint or imprint in re.sub(r"[\s-]", "", p["imprint"]))
    ]
    return jsonify({"success": True, "matches": matches})


def _reasoning_options(model):
    """Keep 'thinking' models quick and make sure the answer comes back as plain text."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    return {}


def ai_complete(messages, vision=False, temperature=0.3, max_tokens=1000):
    """Ask Groq, trying each model in turn. Returns the answer text or raises the last error."""
    last_error = None
    for model in (VISION_MODELS if vision else TEXT_MODELS):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            extra = _reasoning_options(model)
            if extra:
                kwargs["extra_body"] = extra
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if text:
                return text
            last_error = RuntimeError(f"{model} returned an empty answer")
        except Exception as exc:
            last_error = exc
            print(f"[error] Groq model {model} failed: {exc}")
            if getattr(exc, "status_code", None) == 401:
                break  # a bad key fails for every model
    raise last_error or RuntimeError("No AI model is configured.")


def ai_error_message(exc):
    """Turn a Groq error into something a person can act on."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status == 401:
        return "Groq rejected the API key. Check GROQ_API_KEY in your .env file (no quotes or spaces), then restart app.py."
    if status == 403:
        return "Groq refused the request (403). Check that your Groq account and key are active."
    if status == 429:
        return "Groq's usage limit was reached for the moment. Wait a minute and try again."
    if status in (400, 404):
        return ("The AI model isn't available. Set GROQ_TEXT_MODEL in .env to a current model from "
                "console.groq.com/docs/models, then restart app.py.")
    if "Connection" in name or "Timeout" in name:
        return "The server couldn't connect to Groq. Check this computer's internet connection and try again."
    return f"The AI service had a problem: {str(exc)[:160]}"


@app.route("/api/pill-photo", methods=["POST"])
@require_client
def pill_photo(cid):
    """Pill check: read shape/color/imprint from a camera photo and compare with the user's list."""
    upload = request.files.get("file")
    if not upload:
        return jsonify({"success": False, "error": "No photo received."}), 400
    if not client:
        return jsonify({"success": False, "error": "Pill check needs the AI key (GROQ_API_KEY) on the server."}), 503
    try:
        image_url = encode_image(Image.open(io.BytesIO(upload.read())))
    except Exception:
        return jsonify({"success": False, "error": "That photo could not be read."}), 400
    prompt = (
        "Look at the pill or tablet in this photo. Reply with ONLY a JSON object with keys: "
        '"shape" (one of round, oval, oblong, square, diamond, triangle, hexagon, capsule, or "" if unsure), '
        '"color" (one of white, yellow, pink, red, purple, green, blue, brown, orange, gray, or ""), '
        '"imprint" (letters/numbers printed on it, or ""), "description" (one short sentence). '
        'If there is no pill in the photo, return all empty strings and description "No pill found".'
    )
    try:
        raw = ai_complete([{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}}]}], vision=True, temperature=0.1, max_tokens=400)
    except Exception as exc:
        return jsonify({"success": False, "error": ai_error_message(exc)}), 502
    reading = extract_json(raw)
    if reading is None:
        return jsonify({"success": False, "error": "The AI couldn't read the photo. Try again with better light."}), 502
    pill = clean_pill(reading)
    pill["description"] = clean_text(reading.get("description"), "", 160)

    norm = lambda v: re.sub(r"[\s-]", "", str(v or "").upper())
    meds = mutate(cid, lambda state: deepcopy(state["meds"]))
    matches = []
    for m in meds:
        p = m.get("pill") or {}
        if not (p.get("shape") or p.get("color") or p.get("imprint")):
            continue
        score, total, hits = 0, 0, []
        for field, weight in (("shape", 30), ("color", 30)):
            if pill[field] and p.get(field):
                total += weight
                if pill[field] == p[field]:
                    score += weight
                    hits.append(field)
        if pill["imprint"] and p.get("imprint"):
            total += 40
            a, b = norm(pill["imprint"]), norm(p["imprint"])
            if a == b:
                score += 40
                hits.append("imprint")
            elif a and b and (a in b or b in a):
                score += 20
                hits.append("imprint")
        pct = round(100 * score / total) if total else 0
        matches.append({"id": m["id"], "name": m["name"], "dosage": m["dosage"], "pill": p,
                        "score": pct, "hits": hits})
    matches.sort(key=lambda x: -x["score"])
    return jsonify({"success": True, "reading": pill, "matches": matches[:5]})


def language_instruction(lang):
    name = LANGUAGE_NAMES.get(lang, "English")
    return f"Reply in {name}."


@app.route("/api/interactions", methods=["POST"])
def check_interactions():
    data = request.get_json(silent=True) or {}
    med_list = [clean_text(m, "", 80) for m in (data.get("medications") or [])][:30]
    med_list = [m for m in med_list if m]
    lang = data.get("language", "en")

    if not med_list:
        return jsonify({"success": False, "error": "Add at least one medication first."}), 400
    if not client:
        return jsonify({"success": True, "analysis": "AI checking is not configured on this server. "
                        "Please ask your pharmacist about potential drug interactions."})

    prompt = (
        "You are a clinical pharmacist safety assistant. Review this list of medications: "
        f"{', '.join(med_list)}. Check for major drug interactions, contraindications, and "
        "important safety warnings. Give a concise, professional summary in plain text. "
        f"End with a reminder to confirm with a pharmacist or doctor. {language_instruction(lang)}"
    )
    try:
        analysis = ai_complete([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=1200)
        return jsonify({"success": True, "analysis": analysis})
    except Exception as exc:
        print(f"[error] interactions: {exc}")
        return jsonify({"success": False, "error": ai_error_message(exc)}), 502


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    lang = data.get("language", "en")
    meds = [clean_text(m, "", 80) for m in (data.get("medications") or [])][:30]

    messages = []
    for m in (data.get("messages") or [])[-12:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            content = clean_text(m.get("content"), "", 1000)
            if content:
                messages.append({"role": m["role"], "content": content})
    if not messages or messages[-1]["role"] != "user":
        return jsonify({"success": False, "error": "No question provided."}), 400
    if not client:
        return jsonify({"success": False, "error": "The AI assistant isn't configured on this server (missing GROQ_API_KEY)."}), 503

    system_prompt = (
        "You are MedSnap's friendly medication assistant. Answer questions about medications, "
        "schedules, side effects, and food or timing guidance in short, clear, plain language. "
        "You are not a doctor: never change doses or tell the user to stop a prescription; for those, "
        "or for anything urgent or severe, tell them to contact their pharmacist, doctor, or emergency services. "
        + (f"The user's current medications: {', '.join(meds)}. " if meds else "")
        + language_instruction(lang)
    )
    try:
        answer = ai_complete([{"role": "system", "content": system_prompt}] + messages, temperature=0.3, max_tokens=1000)
        return jsonify({"success": True, "answer": answer})
    except Exception as exc:
        print(f"[error] chat: {exc}")
        return jsonify({"success": False, "error": ai_error_message(exc)}), 502


TRANSLATION_CACHE = {}


@app.route("/api/translate", methods=["POST"])
def translate():
    """Translate short free-text instructions (used for the language toggle and read-aloud)."""
    data = request.get_json(silent=True) or {}
    lang = data.get("language", "en")
    texts = [clean_text(t, "", 300) for t in (data.get("texts") or [])][:30]
    texts = [t for t in texts if t]
    if lang not in LANGUAGE_NAMES or lang == "en" or not texts:
        return jsonify({"success": True, "translations": {t: t for t in texts}})

    result = {t: TRANSLATION_CACHE[(lang, t)] for t in texts if (lang, t) in TRANSLATION_CACHE}
    missing = [t for t in texts if t not in result]
    if missing and client:
        prompt = (
            f"Translate each of these medication instructions into {LANGUAGE_NAMES[lang]}. "
            "Keep medication names and numbers unchanged. Reply with ONLY a JSON array of strings, "
            f"in the same order, with exactly {len(missing)} items.\n" + json.dumps(missing, ensure_ascii=False)
        )
        try:
            raw = ai_complete([{"role": "user", "content": prompt}], temperature=0, max_tokens=1600)
            match = re.search(r"\[.*\]", raw, re.S)
            translated = json.loads(match.group(0)) if match else []
            if isinstance(translated, list) and len(translated) == len(missing):
                for src, dst in zip(missing, translated):
                    dst = clean_text(dst, src, 600)
                    result[src] = dst
                    if len(TRANSLATION_CACHE) < 2000:
                        TRANSLATION_CACHE[(lang, src)] = dst
        except Exception as exc:
            print(f"[error] translate: {exc}")
    for t in texts:
        result.setdefault(t, t)  # fall back to the original text
    return jsonify({"success": True, "translations": result})


def extract_json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def encode_image(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img)  # phone photos often arrive rotated
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.thumbnail((1600, 1600))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    upload = request.files.get("file")
    if not upload or upload.filename == "":
        return jsonify({"success": False, "error": "No prescription or pill bottle image uploaded."}), 400
    if not client:
        return jsonify({"success": False, "error": "AI scanning is not configured on this server (missing GROQ_API_KEY)."}), 503
    try:
        image_url = encode_image(Image.open(io.BytesIO(upload.read())))
    except Exception:
        return jsonify({"success": False, "error": "That file could not be read as an image."}), 400

    prompt = (
        "You are an OCR tool reading a pill bottle or prescription label. Reply with ONLY a JSON "
        'object, no other text, with these keys: "name" (medication name), "dosage" (e.g. "10mg"), '
        '"slot" (one of "Morning", "Afternoon", "Evening", based on the directions), '
        '"directions" (the label directions, short). Use an empty string for anything you cannot read.'
    )
    for _ in (1,):
        try:
            raw = ai_complete([{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}]}], vision=True, temperature=0.1, max_tokens=600)
            parsed = extract_json(raw)
            if parsed:
                parsed = {
                    "name": clean_text(parsed.get("name"), "", 80),
                    "dosage": clean_text(parsed.get("dosage"), "", 60),
                    "slot": normalize_slot(parsed.get("slot")),
                    "directions": clean_text(parsed.get("directions"), "", 200),
                }
            return jsonify({"success": True, "filename": upload.filename, "ocr_result": raw, "parsed": parsed})
        except Exception as exc:
            return jsonify({"success": False, "error": ai_error_message(exc)}), 502
    return jsonify({"success": False, "error": "The AI vision service failed."}), 502


# ==========================================================================
# Error handlers: always return something readable, never a blank page
# ==========================================================================
@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({"success": False, "error": err.message}), err.status


@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "error": "Image is too large (10 MB max)."}), 413


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Not found"}), 404
    return Response("<h1>Page not found</h1><p><a href='/'>Back to MedSnap</a></p>", status=404, mimetype="text/html")


@app.errorhandler(500)
def server_error(_):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Server error"}), 500
    return Response("<h1>Something went wrong</h1><p>Please refresh the page.</p>", status=500, mimetype="text/html")


# ==========================================================================
# Local development only (Render uses gunicorn and never runs this block)
# ==========================================================================
if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"\nMedSnap running at http://127.0.0.1:{port}\n")
    if not os.environ.get("RENDER"):
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass
    app.run(host=host, port=port, debug=False)