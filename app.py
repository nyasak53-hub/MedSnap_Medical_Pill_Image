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
VISION_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GROQ_VISION_MODELS",
        "meta-llama/llama-4-scout-17b-16e-instruct,qwen/qwen3.6-27b",
    ).split(",")
    if m.strip()
]
TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

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
VALID_BADGES = {"with_food", "empty_stomach", "avoid_dairy", "avoid_alcohol",
                "avoid_grapefruit", "before_bed", "with_water"}
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
    for key in ("history", "caregiver", "tz", "best_streak"):
        state.setdefault(key, base[key])
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
def ensure_today(state, today):
    hist = state["history"]
    key = today.isoformat()
    rec = hist.get(key)
    if rec is None:
        rec = hist[key] = {"due": [], "taken": [], "alerted": [], "tries": {}, "used": {}}
    for field, default in (("due", []), ("taken", []), ("alerted", []), ("tries", {}), ("used", {})):
        rec.setdefault(field, deepcopy(default))
    for med in state["meds"]:
        if med["id"] not in rec["due"]:
            rec["due"].append(med["id"])
    if len(hist) > 400:  # keep roughly the last year
        for old in sorted(hist.keys())[:-400]:
            del hist[old]
    return rec


def day_complete(state, day):
    rec = state["history"].get(day.isoformat())
    return bool(rec) and bool(rec["due"]) and set(rec["due"]) <= set(rec["taken"])


def compute_streak(state, today):
    day = today if day_complete(state, today) else today - timedelta(days=1)
    streak = 0
    while day_complete(state, day):
        streak += 1
        day -= timedelta(days=1)
    return streak


def build_view(state, today, extra=None):
    rec = ensure_today(state, today)
    taken = set(rec["taken"])
    alerted = set(rec["alerted"])
    meds = [dict(m, taken=m["id"] in taken, alerted=m["id"] in alerted) for m in state["meds"]]
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
    }
    if extra:
        view.update(extra)
    return view


def api_op(client_id, op):
    """Run op(state, today) inside a saved transaction and return the fresh view."""
    tzname = request_tz()
    today = datetime.now(ZoneInfo(tzname)).date()

    def wrapped(state):
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


@app.route("/api/medications", methods=["POST"])
@require_client
def add_medication(cid):
    data = request.get_json(silent=True) or {}

    def op(state, today):
        name = clean_text(data.get("name"))
        if not name:
            raise ApiError("Medication name is required.")
        if len(state["meds"]) >= 100:
            raise ApiError("Medication limit reached.")
        slot = normalize_slot(data.get("slot"))
        sched = str(data.get("scheduled_time") or "")
        badges = [b for b in (data.get("badges") or []) if b in VALID_BADGES]
        state["meds"].append({
            "id": max([m["id"] for m in state["meds"]], default=0) + 1,
            "name": name,
            "dosage": clean_text(data.get("dosage"), "1 pill", 60),
            "slot": slot,
            "stock": clamp_int(data.get("stock"), 30, 0, 10000),
            "daily_dose": clamp_int(data.get("daily_dose"), 1, 1, 20),
            "food_guide": clean_text(data.get("food_guide"), "", 200),
            "badges": list(dict.fromkeys(badges)),
            "scheduled_time": sched if TIME_RE.match(sched) else SLOT_DEFAULT_TIME[slot],
            "critical": bool(data.get("critical")),
            "pill": clean_pill(data.get("pill")),
        })
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
    def op(state, today):
        med = find_med(state, med_id)
        rec = state["history"][today.isoformat()]
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
        amount = clamp_int(data.get("amount"), 0, 0, 10000)
        if amount <= 0:
            raise ApiError("Enter how many pills you added.")
        med["stock"] = min(10000, med["stock"] + amount)
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
        state["caregiver"] = {
            "enabled": enabled,
            "consent": bool(data.get("consent")),
            "patient_name": clean_text(data.get("patient_name"), "", 60),
            "name": clean_text(data.get("name"), "", 60),
            "email": email,
            "phone": phone,
        }
        return None

    return api_op(cid, op)


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
    pending = []
    with DATA_LOCK:
        store = _read_store()
        changed = False
        for cid, raw in store.items():
            state = migrate_state(raw)
            cg = state["caregiver"]
            if not (cg.get("enabled") and cg.get("consent")):
                continue
            tzname = valid_tz(state.get("tz")) or "UTC"
            now = datetime.now(ZoneInfo(tzname))
            rec = ensure_today(state, now.date())
            for med in state["meds"]:
                if not med.get("critical"):
                    continue
                if med["id"] in rec["taken"] or med["id"] in rec["alerted"]:
                    continue
                if int(rec["tries"].get(str(med["id"]), 0)) >= 3:
                    continue
                hh, mm = (int(x) for x in med["scheduled_time"].split(":"))
                due_at = datetime.combine(now.date(), dtime(hh, mm), tzinfo=ZoneInfo(tzname))
                deadline = due_at + timedelta(minutes=60)
                if deadline <= now < deadline + timedelta(hours=ALERT_WINDOW_HOURS):
                    pending.append((cid, med["id"], med["name"], med["dosage"],
                                    med["scheduled_time"], dict(cg), now.date().isoformat()))
            store[cid] = state
            changed = True
        if changed:
            try:
                _write_store(store)
            except Exception as exc:
                print(f"[warn] Could not save data: {exc}")

    for cid, med_id, name, dosage, sched, caregiver, day_key in pending:
        if not (EMAIL_CONFIGURED or SMS_CONFIGURED) or not global_send_allowed():
            continue
        who = caregiver.get("patient_name") or "A MedSnap user"
        greet = f"Hi {caregiver['name']}, " if caregiver.get("name") else ""
        body = (f"{greet}{who} has not marked {name} ({dosage}) as taken. It was scheduled for "
                f"{sched}. Please check in with them. (Sent by MedSnap because you were listed as a caregiver.)")
        delivered = deliver_alert(caregiver, f"MedSnap: missed dose of {name}", body)

        def record(state, _cid=cid, _id=med_id, _day=day_key, _ok=bool(delivered)):
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
        response = client.chat.completions.create(
            model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=600)
        return jsonify({"success": True, "analysis": response.choices[0].message.content})
    except Exception as exc:
        print(f"[error] interactions: {exc}")
        return jsonify({"success": False, "error": "The AI service could not be reached. Try again shortly."}), 502


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
        response = client.chat.completions.create(
            model=TEXT_MODEL, messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.3, max_tokens=500)
        return jsonify({"success": True, "answer": response.choices[0].message.content})
    except Exception as exc:
        print(f"[error] chat: {exc}")
        return jsonify({"success": False, "error": "The AI service could not be reached. Try again shortly."}), 502


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
            response = client.chat.completions.create(
                model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=1200)
            raw = response.choices[0].message.content or ""
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
    for model in VISION_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}]}],
                temperature=0.1, max_tokens=500)
            raw = response.choices[0].message.content or ""
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
            print(f"[error] vision model {model} failed: {exc}")
    return jsonify({"success": False, "error": "The AI vision service failed. Check the Render logs and the GROQ_VISION_MODELS setting."}), 502


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