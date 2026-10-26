import os
import io
import re
import json
import time
import base64
import hashlib
import secrets
import smtplib
import shutil
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
from flask import Flask, request, jsonify, send_from_directory, Response, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageOps
from werkzeug.security import check_password_hash, generate_password_hash

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
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "zh": "Simplified Chinese",
                  "hi": "Hindi (Devanagari script)", "te": "Telugu (Telugu script)"}
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


class StorageBusy(Exception):
    """A saved-data file exists but couldn't be read right now. Never treat it as empty."""


# ==========================================================================
# Permanent storage in a free Postgres database (e.g. Neon) when DATABASE_URL is set.
# Render's free plan erases saved files on every restart, so accounts must live in a database there.
# Without DATABASE_URL (e.g. on your own computer) everything is saved in files next to app.py.
# ==========================================================================
def _clean_database_url(raw):
    """Accept the link however it was pasted: with quotes, with "psql '...'" from Neon's
    Connect box, or with "DATABASE_URL=" in front. Returns (url, problem)."""
    if raw is None or not str(raw).strip():
        return None, None
    m = re.search(r"postgres(?:ql)?://[^\s'\"]+", str(raw))
    if not m:
        return None, ("DATABASE_URL doesn't look like a database link. It must start with postgresql:// "
                      "(copy it from Neon > Connect, and paste only the link).")
    url = m.group(0)
    if "@" not in url:
        return None, "DATABASE_URL is missing the user name and password part (user:password@). Copy the full link from Neon again."
    return url, None


DATABASE_URL, DATABASE_PROBLEM = _clean_database_url(os.environ.get("DATABASE_URL"))
DATABASE_STATUS = "not used" if not DATABASE_URL else "not checked yet"
_pg = None
_DB_HTTP = None        # Neon: talk to the database over normal HTTPS (no extra driver needed)
if DATABASE_URL:
    from urllib.parse import urlsplit
    _db_host = (urlsplit(DATABASE_URL).hostname or "").lower()
    _http_endpoint = os.environ.get("NEON_HTTP_ENDPOINT")       # only used for testing
    if _http_endpoint or _db_host.endswith(".neon.tech") or _db_host.endswith(".neon.build"):
        _DB_HTTP = _http_endpoint or f"https://{_db_host}/sql"
    else:
        try:
            import psycopg as _pg
        except ImportError:
            DATABASE_URL = None
            DATABASE_PROBLEM = ("this database needs the psycopg driver. Add this line to requirements.txt "
                                "and push again:  psycopg[binary]   (Neon databases don't need it.)")
if DATABASE_PROBLEM:
    print("\n" + "!" * 78 + f"\n[VeraDose] DATABASE PROBLEM: {DATABASE_PROBLEM}\n"
          "VeraDose is running, but accounts are NOT being saved permanently until this is fixed.\n" + "!" * 78 + "\n")

_KV_TABLE_SQL = ("CREATE TABLE IF NOT EXISTS medsnap_kv ("
                 "k TEXT PRIMARY KEY, v TEXT NOT NULL, updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)")
_db_conn = None
_db_lock = threading.Lock()
_db_table_ready = False


class _DbHttpError(Exception):
    pass


def _db_http(sql, params=()):
    """One query over HTTPS to Neon (POST https://<host>/sql), using only Python's built-in
    tools. Each call is independent and has a hard time limit, so a slow or sleeping
    database can never freeze the website."""
    import urllib.request
    import urllib.error
    n = 0

    def number(_m):
        nonlocal n
        n += 1
        return f"${n}"
    query = re.sub(r"%s", number, sql)
    body = json.dumps({"query": query, "params": [str(p) for p in params]}).encode("utf-8")
    req = urllib.request.Request(_DB_HTTP, data=body, method="POST", headers={
        "Content-Type": "application/json", "Neon-Connection-String": DATABASE_URL})
    try:
        with urllib.request.urlopen(req, timeout=12) as res:
            data = json.loads(res.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        try:
            msg = json.loads(exc.read().decode("utf-8") or "{}").get("message")
        except Exception:
            msg = None
        raise _DbHttpError(msg or f"database answered with HTTP {exc.code}")
    except Exception as exc:   # no internet, wrong address, timeout, bad reply...
        raise _DbHttpError(f"could not reach the database ({exc.__class__.__name__}: {exc})")
    rows = data.get("rows") or []
    if not rows:
        return None
    row = rows[0]
    return tuple(row.values()) if isinstance(row, dict) else tuple(row)


def _db_connect():
    global _db_conn
    # keepalives + tcp_user_timeout: a dropped connection fails in seconds instead of hanging the site
    _db_conn = _pg.connect(DATABASE_URL, autocommit=True, connect_timeout=8,
                           keepalives=1, keepalives_idle=20, keepalives_interval=5, keepalives_count=3,
                           tcp_user_timeout=10000)
    _db_conn.execute(_KV_TABLE_SQL)


def _db(sql, params=()):
    """Run one query. Neon goes over HTTPS; other Postgres databases use a shared psycopg
    connection that reconnects once if the database closed it."""
    global _db_conn, DATABASE_STATUS, _db_table_ready
    if _DB_HTTP:
        last = None
        for attempt in (1, 2):
            try:
                if not _db_table_ready:
                    _db_http(_KV_TABLE_SQL)
                    _db_table_ready = True
                out = _db_http(sql, params)
                DATABASE_STATUS = "connected"
                return out
            except _DbHttpError as exc:
                last = exc
                DATABASE_STATUS = "error: " + _db_hint(exc)
                print(f"[warn] Database problem ({_db_hint(exc)}).")
                if "password" in str(exc).lower() or "does not exist" in str(exc).lower():
                    break   # retrying won't help
        raise StorageBusy(str(last))
    # Never let requests pile up behind a stuck database call.
    if not _db_lock.acquire(timeout=25):
        raise StorageBusy("database busy")
    try:
        for attempt in (1, 2):
            try:
                if _db_conn is None or getattr(_db_conn, "closed", False):
                    _db_connect()
                cur = _db_conn.execute(sql, params)
                DATABASE_STATUS = "connected"
                return cur.fetchone() if cur.description else None
            except _pg.Error as exc:
                DATABASE_STATUS = "error: " + _db_hint(exc)
                print(f"[warn] Database problem ({_db_hint(exc)}); reconnecting.")
                try:
                    _db_conn.close()
                except Exception:
                    pass
                _db_conn = None
                if attempt == 2:
                    raise StorageBusy(str(exc))
    finally:
        _db_lock.release()


def _db_hint(exc):
    """Explain a database error in plain words (never prints the password)."""
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    low = msg.lower()
    if "password authentication failed" in low:
        tip = "the password in DATABASE_URL is wrong. Copy the link from Neon again (click Show password first)."
    elif "could not translate host name" in low or "name or service not known" in low or "nodename nor servname" in low:
        tip = "the database address in DATABASE_URL is wrong or cut off. Copy the full link from Neon again."
    elif "does not exist" in low and "database" in low:
        tip = "the database name at the end of DATABASE_URL doesn't exist. Copy the link from Neon again."
    elif "invalid dsn" in low or "missing \"=\"" in low or "invalid connection option" in low:
        tip = "DATABASE_URL has extra text in it. Paste only the link that starts with postgresql://"
    elif ("could not reach the database" in low or "connecterror" in low) and "timeout" not in low:
        tip = "VeraDose couldn't reach the database address in DATABASE_URL. Copy the full link from Neon again."
    elif "timeout" in low or "timed out" in low:
        tip = "the database took too long to answer (it may be waking up). It will try again."
    else:
        tip = msg
    return re.sub(r"://[^@\s]+@", "://***@", tip)


def _check_database_at_startup():
    global DATABASE_STATUS
    try:
        _db("SELECT 1")
        DATABASE_STATUS = "connected"
        print("[VeraDose] Database connected. Accounts and medicines are saved permanently.")
    except Exception as exc:
        DATABASE_STATUS = "error: " + _db_hint(exc)
        print("\n" + "!" * 78 + f"\n[VeraDose] COULD NOT CONNECT TO THE DATABASE: {_db_hint(exc)}\n"
              "The website still opens, but signing in won't work until this is fixed.\n" + "!" * 78 + "\n")


def _db_key(path):
    return os.path.basename(path)          # "users.json" or "medications_data.json"


def _db_load(path):
    row = _db("SELECT v FROM medsnap_kv WHERE k = %s", (_db_key(path),))
    if row:
        return json.loads(row[0])
    # First run with a database: bring over anything already saved in the local file.
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _db_save(path, data)
                print(f"[VeraDose] Copied {os.path.basename(path)} into the database.")
                return data
        except Exception as exc:
            print(f"[warn] Could not import {path}: {exc}")
    return None


def _db_save(path, data):
    _db("INSERT INTO medsnap_kv (k, v, updated_at) VALUES (%s, %s, CURRENT_TIMESTAMP) "
        "ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v, updated_at = CURRENT_TIMESTAMP",
        (_db_key(path), json.dumps(data)))


def storage_mode():
    return "database" if DATABASE_URL else "files"


def _load_json(path):
    if DATABASE_URL:
        return _db_load(path)
    return _load_json_file(path)


def _load_json_file(path):
    """Read a saved JSON file safely.
    - Missing file: returns None (nothing saved yet).
    - Briefly locked (OneDrive, antivirus, backup tools): retries for a moment.
    - Damaged: falls back to the automatic .bak copy.
    It never returns "empty" for a file that exists, so saved accounts can't be wiped by a failed read."""
    backup = path + ".bak"
    if not os.path.exists(path) and not os.path.exists(backup):
        return None
    last = None
    for attempt in range(6):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            last = ValueError("not a JSON object")
            break
        except FileNotFoundError:
            break
        except PermissionError as exc:
            last = exc
            time.sleep(0.15)
        except (OSError, ValueError) as exc:
            last = exc
            break
    if os.path.exists(backup):
        try:
            with open(backup, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                print(f"[warn] {os.path.basename(path)} could not be read ({last}); using the backup copy.")
                return data
        except Exception:
            pass
    if not os.path.exists(path):
        return None
    print(f"[error] {path} could not be read ({last}). Nothing was overwritten.")
    raise StorageBusy(str(last))


def _save_json(path, data):
    if DATABASE_URL:
        _db_save(path, data)
        return
    _save_json_file(path, data)


def _save_json_file(path, data):
    """Write atomically (never half-written) and keep the previous version as a .bak copy."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=folder, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        if os.path.exists(path):
            try:
                shutil.copyfile(path, path + ".bak")
            except OSError:
                pass
        for attempt in range(6):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.15)   # file briefly locked by sync/antivirus software
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_store():
    data = _load_json(DATA_FILE)
    return data if data is not None else {}


def _write_store(store):
    _save_json(DATA_FILE, store)


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
                raise StorageBusy(str(exc))
        return result


# ==========================================================================
# Accounts (sign up / sign in). Each account owns one saved state in the data file.
# ==========================================================================
USERS_FILE = os.path.join(DATA_DIR, "users.json")
USERS_LOCK = threading.RLock()
LOGIN_LOG = {}


def _read_users():
    data = _load_json(USERS_FILE) or {}
    data.setdefault("users", {})
    data.setdefault("tokens", {})
    return data


def _write_users(data):
    _save_json(USERS_FILE, data)


def _token_key(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_token(data, email):
    token = secrets.token_urlsafe(32)
    data["tokens"][_token_key(token)] = {"email": email, "created": time.time()}
    return token


def account_for_token(token):
    if not token or len(token) > 200:
        return None
    with USERS_LOCK:
        data = _read_users()
        entry = data["tokens"].get(_token_key(token))
        if not entry:
            return None
        return data["users"].get(entry["email"])


AUTH_COOKIE = "medsnap_token"
REMEMBER_SECONDS = 400 * 24 * 3600  # stay signed in (browsers cap cookies at about 400 days)


def request_tokens():
    """The sign-in token can come from the page (X-Auth-Token) or the remember-me cookie."""
    return [t for t in (request.headers.get("X-Auth-Token", ""), request.cookies.get(AUTH_COOKIE, "")) if t]


def account_for_request():
    for tok in request_tokens():
        account = account_for_token(tok)
        if account:
            return account
    return None


def is_https():
    return request.headers.get("X-Forwarded-Proto", request.scheme) == "https"


def signed_in_response(payload, token):
    """Reply to sign up / sign in and set a long-lasting cookie so the person stays signed in."""
    resp = make_response(jsonify(payload))
    resp.set_cookie(AUTH_COOKIE, token, max_age=REMEMBER_SECONDS, httponly=True, samesite="Lax",
                    secure=is_https(), path="/")
    return resp


def signed_out_response(payload):
    resp = make_response(jsonify(payload))
    resp.delete_cookie(AUTH_COOKIE, path="/")
    return resp


def login_allowed(ip, limit=10):
    """At most 10 sign-in/sign-up attempts per 15 minutes from one address."""
    now = time.time()
    stamps = [t for t in LOGIN_LOG.get(ip, []) if now - t < 900]
    if len(stamps) >= limit:
        LOGIN_LOG[ip] = stamps
        return False
    stamps.append(now)
    LOGIN_LOG[ip] = stamps
    return True


def require_client(fn):
    """Every data request must come from a signed-in account."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        account = account_for_request()
        if not account:
            return jsonify({"success": False, "error": "Please sign in to continue.", "auth": False}), 401
        return fn(account["cid"], *args, **kwargs)
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
            "<h1>VeraDose is running</h1><p>The server started, but <code>index.html</code> was not "
            "found next to <code>app.py</code>. Make sure it is committed and pushed to GitHub.</p>",
            status=500, mimetype="text/html")
    )


@app.route("/app")
def main_app():
    return page_response("index.html") or landing()


# ==========================================================================
# Installable app (PWA): manifest, icons, service worker, install helper
# Everything is generated here, so no extra files or folders are needed.
# ==========================================================================
PWA_VERSION = "veradose-v5"
BRAND = (59, 98, 85)          # #3B6255, the VeraDose green
BRAND_LIGHT = (203, 222, 211)  # #CBDED3


def _make_icon(size, maskable=False, rounded=True):
    """Draw the VeraDose icon: a white-and-mint capsule on the brand green."""
    from PIL import ImageDraw
    s = 4  # draw big, then shrink, for smooth edges
    W = size * s
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if maskable or not rounded:
        d.rectangle([0, 0, W, W], fill=BRAND + (255,))
    else:
        d.rounded_rectangle([0, 0, W - 1, W - 1], radius=int(W * 0.22), fill=BRAND + (255,))
    # capsule, drawn flat then rotated 45 degrees
    area = 0.62 if maskable else 0.70          # maskable icons keep a safe margin
    cw, ch = int(W * area * 0.98), int(W * area * 0.40)
    cap = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    cd = ImageDraw.Draw(cap)
    r = ch // 2
    cd.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=r, fill=(255, 255, 255, 255))
    half = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    hd = ImageDraw.Draw(half)
    hd.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=r, fill=BRAND_LIGHT + (255,))
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).rectangle([cw // 2, 0, cw, ch], fill=255)
    cap.paste(half, (0, 0), Image.composite(half.split()[3], mask, mask))
    line = max(2, W // 90)
    cd.line([(cw // 2, 0), (cw // 2, ch)], fill=BRAND + (255,), width=line)
    cap = cap.rotate(45, resample=Image.BICUBIC, expand=True)
    img.alpha_composite(cap, ((W - cap.width) // 2, (W - cap.height) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    if not rounded and not maskable:
        img = img.convert("RGB")  # Apple adds its own rounded corners
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


_ICON_SPECS = {
    "icon-192.png": (192, False, True),
    "icon-512.png": (512, False, True),
    "maskable-512.png": (512, True, False),
    "apple-touch-icon.png": (180, False, False),
    "favicon-32.png": (32, False, True),
}
_icon_cache = {}


@app.route("/icons/<name>")
def pwa_icon(name):
    if name not in _ICON_SPECS:
        return Response("Not found", status=404)
    if name not in _icon_cache:
        _icon_cache[name] = _make_icon(*_ICON_SPECS[name])
    resp = Response(_icon_cache[name], mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=604800"
    return resp


@app.route("/apple-touch-icon.png")
@app.route("/apple-touch-icon-precomposed.png")
def apple_touch_icon():
    return pwa_icon("apple-touch-icon.png")


@app.route("/favicon.ico")
def favicon():
    return pwa_icon("favicon-32.png")


@app.route("/manifest.webmanifest")
def pwa_manifest():
    manifest = {
        "id": "/app",
        "name": "VeraDose",
        "short_name": "VeraDose",
        "description": "Medication reminders, pill check and a friendly AI helper.",
        "start_url": "/app",
        "scope": "/",
        "display": "standalone",
        "orientation": "any",
        "background_color": "#E9F1EC",
        "theme_color": "#3B6255",
        "categories": ["health", "medical", "lifestyle"],
        "icons": [
            {"src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
            {"src": "/icons/maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        ],
        "shortcuts": [
            {"name": "Today", "url": "/app#today", "icons": [{"src": "/icons/icon-192.png", "sizes": "192x192"}]},
            {"name": "Pill check", "url": "/app#pillcheck", "icons": [{"src": "/icons/icon-192.png", "sizes": "192x192"}]},
            {"name": "Calendar", "url": "/app#calendar", "icons": [{"src": "/icons/icon-192.png", "sizes": "192x192"}]},
        ],
    }
    resp = Response(json.dumps(manifest), mimetype="application/manifest+json")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


SERVICE_WORKER_JS = r"""
// VeraDose service worker: lets the installed app open fast and work offline.
// Your data (/api/...) is never stored here; it always comes from the server.
const CACHE = '__VERSION__';
const CORE = ['/', '/app', '/pwa.js', '/manifest.webmanifest', '/icons/icon-192.png', '/icons/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => Promise.all(CORE.map((u) => c.add(u).catch(() => {})))).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

function offlinePage() {
  return new Response('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<title>VeraDose</title><body style="font-family:system-ui,sans-serif;background:#E9F1EC;color:#1F3A30;display:grid;place-items:center;min-height:100vh;margin:0;text-align:center;padding:1rem">' +
    '<div><h1 style="color:#3B6255">VeraDose</h1><p>You are offline. Connect to the internet and try again.</p>' +
    '<button onclick="location.reload()" style="font:inherit;padding:.7rem 1.4rem;border:0;border-radius:999px;background:#3B6255;color:#fff">Try again</button></div>',
    { headers: { 'Content-Type': 'text/html; charset=utf-8' } });
}

// Tapping an alarm notification opens (or brings back) VeraDose.
self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  const url = (e.notification.data && e.notification.data.url) || '/app#alarms';
  e.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((list) => {
    for (const c of list) { if (new URL(c.url).pathname.startsWith('/app')) { c.focus(); return; } }
    return self.clients.openWindow(url);
  }));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;          // fonts, icons from other sites: browser handles them
  if (url.pathname.startsWith('/api/') || url.pathname === '/healthz' || url.pathname === '/sw.js') return;

  if (req.mode === 'navigate') {
    // Pages: try the internet first. If the server is slow to wake up (Render free),
    // show the saved copy after 4 seconds; the page then reconnects by itself.
    const key = url.pathname === '/' ? '/' : (url.pathname.startsWith('/app') ? '/app' : url.pathname);
    e.respondWith(new Promise((resolve) => {
      let done = false;
      const fromCache = () => caches.match(key).then((r) => r || caches.match(req));
      const net = fetch(req).then((res) => {
        if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(key, copy)); }
        if (!done) { done = true; resolve(res); }
      }).catch(() => fromCache().then((r) => { if (!done) { done = true; resolve(r || offlinePage()); } }));
      setTimeout(() => fromCache().then((r) => { if (r && !done) { done = true; resolve(r); } }), 4000);
      e.waitUntil(net.catch(() => {}));
    }));
    return;
  }

  // Small files (icons, pwa.js, manifest): show the saved copy, refresh it in the background.
  e.respondWith(caches.match(req).then((hit) => {
    const net = fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    }).catch(() => hit);
    return hit || net;
  }));
});
"""


@app.route("/sw.js")
def service_worker():
    resp = Response(SERVICE_WORKER_JS.replace("__VERSION__", PWA_VERSION), mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


PWA_JS = r"""
/* VeraDose install helper: registers the service worker and runs the "Install app" buttons.
   Any element with data-install becomes an install button. It hides itself once installed. */
(function () {
  var TXT = {
    en: { btn: 'Install app', title: 'Install VeraDose', sub: 'Add VeraDose to your home screen. It opens like a normal app, for free.',
          ios: ['Tap the <b>Share</b> button (the square with an arrow) at the bottom or top of Safari.', 'Scroll down and tap <b>Add to Home Screen</b>.', 'Tap <b>Add</b>. VeraDose is now on your home screen.'],
          iosOther: 'On iPhone and iPad, open this page in <b>Safari</b> first, then follow these steps.',
          android: ['Tap the <b>⋮</b> menu at the top right of your browser.', 'Tap <b>Install app</b> or <b>Add to Home screen</b>.', 'Tap <b>Install</b>. VeraDose is now on your home screen.'],
          desktop: ['In <b>Chrome</b> or <b>Edge</b>, click the install icon at the right end of the address bar, or open the <b>⋮</b> menu and choose <b>Install VeraDose</b>.', 'In <b>Safari</b> on a Mac, choose <b>File → Add to Dock</b>.', 'Firefox on a computer cannot install apps. Please use Chrome, Edge or Safari.'],
          ok: 'Got it', done: 'VeraDose is installed! Look for it on your home screen.' },
    es: { btn: 'Instalar app', title: 'Instalar VeraDose', sub: 'Añade VeraDose a tu pantalla de inicio. Se abre como una app normal, gratis.',
          ios: ['Toca el botón <b>Compartir</b> (el cuadrado con una flecha) en Safari.', 'Desliza hacia abajo y toca <b>Agregar a inicio</b>.', 'Toca <b>Agregar</b>. VeraDose ya está en tu pantalla de inicio.'],
          iosOther: 'En iPhone y iPad, primero abre esta página en <b>Safari</b> y luego sigue estos pasos.',
          android: ['Toca el menú <b>⋮</b> arriba a la derecha del navegador.', 'Toca <b>Instalar aplicación</b> o <b>Agregar a la pantalla principal</b>.', 'Toca <b>Instalar</b>. VeraDose ya está en tu pantalla de inicio.'],
          desktop: ['En <b>Chrome</b> o <b>Edge</b>, haz clic en el icono de instalar al final de la barra de direcciones, o abre el menú <b>⋮</b> y elige <b>Instalar VeraDose</b>.', 'En <b>Safari</b> en Mac, elige <b>Archivo → Añadir al Dock</b>.', 'Firefox en computadora no puede instalar apps. Usa Chrome, Edge o Safari.'],
          ok: 'Entendido', done: '¡VeraDose está instalada! Búscala en tu pantalla de inicio.' },
    fr: { btn: "Installer l'app", title: 'Installer VeraDose', sub: "Ajoutez VeraDose à votre écran d'accueil. Elle s'ouvre comme une vraie app, gratuitement.",
          ios: ['Touchez le bouton <b>Partager</b> (le carré avec une flèche) dans Safari.', "Faites défiler et touchez <b>Sur l'écran d'accueil</b>.", "Touchez <b>Ajouter</b>. VeraDose est sur votre écran d'accueil."],
          iosOther: "Sur iPhone et iPad, ouvrez d'abord cette page dans <b>Safari</b>, puis suivez ces étapes.",
          android: ['Touchez le menu <b>⋮</b> en haut à droite du navigateur.', "Touchez <b>Installer l'application</b> ou <b>Ajouter à l'écran d'accueil</b>.", "Touchez <b>Installer</b>. VeraDose est sur votre écran d'accueil."],
          desktop: ["Dans <b>Chrome</b> ou <b>Edge</b>, cliquez sur l'icône d'installation au bout de la barre d'adresse, ou ouvrez le menu <b>⋮</b> et choisissez <b>Installer VeraDose</b>.", 'Dans <b>Safari</b> sur Mac, choisissez <b>Fichier → Ajouter au Dock</b>.', 'Firefox sur ordinateur ne peut pas installer d’apps. Utilisez Chrome, Edge ou Safari.'],
          ok: "J'ai compris", done: "VeraDose est installée ! Cherchez-la sur votre écran d'accueil." },
    zh: { btn: '安装应用', title: '安装 VeraDose', sub: '把 VeraDose 添加到主屏幕，像普通应用一样打开，完全免费。',
          ios: ['在 Safari 中点按<b>分享</b>按钮（带箭头的方框）。', '向下滑动，点按<b>添加到主屏幕</b>。', '点按<b>添加</b>。VeraDose 已在你的主屏幕上。'],
          iosOther: '在 iPhone 和 iPad 上，请先用 <b>Safari</b> 打开此页面，再按以下步骤操作。',
          android: ['点按浏览器右上角的 <b>⋮</b> 菜单。', '点按<b>安装应用</b>或<b>添加到主屏幕</b>。', '点按<b>安装</b>。VeraDose 已在你的主屏幕上。'],
          desktop: ['在 <b>Chrome</b> 或 <b>Edge</b> 中，点击地址栏右侧的安装图标，或打开 <b>⋮</b> 菜单选择<b>安装 VeraDose</b>。', '在 Mac 的 <b>Safari</b> 中，选择<b>文件 → 添加到程序坞</b>。', '电脑上的 Firefox 无法安装应用，请使用 Chrome、Edge 或 Safari。'],
          ok: '知道了', done: 'VeraDose 已安装！请在主屏幕上找到它。' },
    hi: { btn: 'ऐप इंस्टॉल करें', title: 'VeraDose इंस्टॉल करें', sub: 'VeraDose को अपनी होम स्क्रीन पर जोड़ें। यह सामान्य ऐप की तरह खुलता है, मुफ़्त में।',
          ios: ['Safari में <b>शेयर</b> बटन (तीर वाला चौकोर) टैप करें।', 'नीचे स्क्रॉल करें और <b>होम स्क्रीन में जोड़ें</b> टैप करें।', '<b>जोड़ें</b> टैप करें। VeraDose अब आपकी होम स्क्रीन पर है।'],
          iosOther: 'iPhone और iPad पर पहले यह पेज <b>Safari</b> में खोलें, फिर ये चरण अपनाएँ।',
          android: ['ब्राउज़र में ऊपर दाईं ओर <b>⋮</b> मेनू टैप करें।', '<b>ऐप इंस्टॉल करें</b> या <b>होम स्क्रीन में जोड़ें</b> टैप करें।', '<b>इंस्टॉल करें</b> टैप करें। VeraDose अब आपकी होम स्क्रीन पर है।'],
          desktop: ['<b>Chrome</b> या <b>Edge</b> में पता बार के दाईं ओर इंस्टॉल आइकन पर क्लिक करें, या <b>⋮</b> मेनू खोलकर <b>VeraDose इंस्टॉल करें</b> चुनें।', 'Mac पर <b>Safari</b> में <b>File → Add to Dock</b> चुनें।', 'कंप्यूटर पर Firefox ऐप इंस्टॉल नहीं कर सकता। कृपया Chrome, Edge या Safari इस्तेमाल करें।'],
          ok: 'समझ गया', done: 'VeraDose इंस्टॉल हो गया! इसे अपनी होम स्क्रीन पर देखें।' },
    te: { btn: 'యాప్ ఇన్‌స్టాల్ చేయండి', title: 'VeraDose ఇన్‌స్టాల్ చేయండి', sub: 'VeraDose ను మీ హోమ్ స్క్రీన్‌కు జోడించండి. ఇది సాధారణ యాప్‌లా తెరుచుకుంటుంది, ఉచితంగా.',
          ios: ['Safari లో <b>షేర్</b> బటన్ (బాణం ఉన్న చతురస్రం) నొక్కండి.', 'కిందికి స్క్రోల్ చేసి <b>హోమ్ స్క్రీన్‌కు జోడించు</b> నొక్కండి.', '<b>జోడించు</b> నొక్కండి. VeraDose ఇప్పుడు మీ హోమ్ స్క్రీన్‌పై ఉంది.'],
          iosOther: 'iPhone మరియు iPad లో ముందుగా ఈ పేజీని <b>Safari</b> లో తెరిచి, తర్వాత ఈ దశలు పాటించండి.',
          android: ['బ్రౌజర్ పైభాగంలో కుడివైపు <b>⋮</b> మెనూ నొక్కండి.', '<b>యాప్ ఇన్‌స్టాల్ చేయండి</b> లేదా <b>హోమ్ స్క్రీన్‌కు జోడించు</b> నొక్కండి.', '<b>ఇన్‌స్టాల్</b> నొక్కండి. VeraDose ఇప్పుడు మీ హోమ్ స్క్రీన్‌పై ఉంది.'],
          desktop: ['<b>Chrome</b> లేదా <b>Edge</b> లో అడ్రస్ బార్ కుడి చివర ఉన్న ఇన్‌స్టాల్ ఐకాన్ నొక్కండి, లేదా <b>⋮</b> మెనూ తెరిచి <b>VeraDose ఇన్‌స్టాల్ చేయండి</b> ఎంచుకోండి.', 'Mac లో <b>Safari</b> లో <b>File → Add to Dock</b> ఎంచుకోండి.', 'కంప్యూటర్‌లో Firefox యాప్‌లను ఇన్‌స్టాల్ చేయలేదు. దయచేసి Chrome, Edge లేదా Safari వాడండి.'],
          ok: 'అర్థమైంది', done: 'VeraDose ఇన్‌స్టాల్ అయింది! మీ హోమ్ స్క్రీన్‌పై చూడండి.' }
  };
  function L() { var l = 'en'; try { l = localStorage.getItem('medsnap_lang') || document.documentElement.lang || 'en'; } catch (e) {} l = String(l).slice(0, 2); return TXT[l] || TXT.en; }
  var root = document.documentElement;
  var deferred = null;
  var ua = navigator.userAgent || '';
  var isIOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  var isSafari = /Safari/.test(ua) && !/CriOS|FxiOS|EdgiOS|OPiOS|Chrome|Android/.test(ua);
  var isAndroid = /Android/.test(ua);
  function standalone() { return (window.matchMedia && matchMedia('(display-mode: standalone)').matches) || navigator.standalone === true; }

  function refresh() {
    if (standalone()) { root.classList.remove('can-install'); root.classList.add('is-installed'); }
    else root.classList.add('can-install');
  }

  if ('serviceWorker' in navigator && /^https?:$/.test(location.protocol)) {
    window.addEventListener('load', function () { navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(function () {}); });
  }
  window.addEventListener('beforeinstallprompt', function (e) { e.preventDefault(); deferred = e; refresh(); });
  window.addEventListener('appinstalled', function () { deferred = null; root.classList.remove('can-install'); toast(L().done); });

  function toast(msg) {
    var d = document.createElement('div');
    d.textContent = msg; d.setAttribute('role', 'status');
    d.style.cssText = 'position:fixed;left:50%;bottom:1.4rem;transform:translateX(-50%);z-index:9999;background:#3B6255;color:#fff;padding:.8rem 1.2rem;border-radius:999px;font:600 1rem system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.2);max-width:calc(100vw - 2rem);text-align:center';
    document.body.appendChild(d); setTimeout(function () { d.remove(); }, 4500);
  }

  function help() {
    var t = L(), steps, note = '';
    if (isIOS) { steps = t.ios; if (!isSafari) note = t.iosOther; }
    else if (isAndroid) steps = t.android;
    else steps = t.desktop;
    var dark = root.getAttribute('data-theme') === 'dark';
    var bg = dark ? '#17261F' : '#FFFFFF', ink = dark ? '#CBDED3' : '#1F3A30', mute = dark ? '#8BA49A' : '#3B6255';
    var dlg = document.createElement('dialog');
    dlg.setAttribute('aria-labelledby', 'pwaTitle');
    dlg.style.cssText = 'margin:auto;inset:0;position:fixed;max-height:calc(100vh - 2rem);overflow:auto;border:0;border-radius:1.4rem;padding:0;width:min(28rem,calc(100vw - 2rem));background:' + bg + ';color:' + ink + ';box-shadow:0 20px 60px rgba(0,0,0,.3);font-family:inherit';
    dlg.innerHTML = '<div style="padding:1.5rem 1.5rem 1.2rem">' +
      '<div style="display:flex;gap:.9rem;align-items:center;margin-bottom:.6rem"><img src="/icons/icon-192.png" alt="" width="52" height="52" style="border-radius:.9rem">' +
      '<h2 id="pwaTitle" style="margin:0;font-size:1.35rem">' + t.title + '</h2></div>' +
      '<p style="margin:.2rem 0 1rem;color:' + mute + '">' + t.sub + '</p>' +
      (note ? '<p style="margin:0 0 .8rem;padding:.7rem .9rem;border-radius:.8rem;background:' + (dark ? '#22362C' : '#DCE7DF') + '">' + note + '</p>' : '') +
      '<ol style="margin:0 0 1.2rem;padding-left:1.3rem;display:grid;gap:.55rem;line-height:1.45">' + steps.map(function (s) { return '<li>' + s + '</li>'; }).join('') + '</ol>' +
      '<button type="button" style="width:100%;font:inherit;font-weight:700;padding:.8rem;border:0;border-radius:999px;background:#3B6255;color:#fff;cursor:pointer">' + t.ok + '</button></div>';
    document.body.appendChild(dlg);
    dlg.querySelector('button').addEventListener('click', function () { dlg.close(); });
    dlg.addEventListener('close', function () { dlg.remove(); });
    dlg.addEventListener('click', function (e) { if (e.target === dlg) dlg.close(); });
    if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', '');
  }

  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-install]');
    if (!b) return;
    e.preventDefault();
    if (deferred) {
      var p = deferred; deferred = null;
      p.prompt();
      p.userChoice.then(function () { refresh(); }).catch(function () {});
    } else help();
  });

  window.VeraDosePWA = { refresh: refresh, help: help };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', refresh); else refresh();
})();
"""


@app.route("/pwa.js")
def pwa_js():
    resp = Response(PWA_JS, mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "index_html_found": os.path.isfile(os.path.join(BASE_DIR, "index.html")),
        "landing_html_found": os.path.isfile(os.path.join(BASE_DIR, "landing.html")),
        "ai_enabled": client is not None,
        "email_alerts": EMAIL_CONFIGURED,
        "sms_alerts": SMS_CONFIGURED,
        "storage": storage_mode(),
        "database": DATABASE_STATUS if not DATABASE_PROBLEM else "problem: " + DATABASE_PROBLEM,
        "accounts_saved_permanently": bool(DATABASE_URL) or not os.environ.get("RENDER"),
    })


# ==========================================================================
# Medication API
# ==========================================================================
@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    if not login_allowed(client_ip()):
        return jsonify({"success": False, "error": "Too many attempts. Please wait a few minutes."}), 429
    data = request.get_json(silent=True) or {}
    name = clean_text(data.get("name"), "", 60)
    email = clean_text(data.get("email"), "", 254).lower()
    password = str(data.get("password") or "")
    age = str(data.get("age") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "Please enter your name."}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"success": False, "error": "Please enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"success": False, "error": "Your password needs at least 8 characters."}), 400
    if age and not (age.isdigit() and 1 <= int(age) <= 120):
        return jsonify({"success": False, "error": "Age should be a number between 1 and 120."}), 400
    with USERS_LOCK:
        users = _read_users()
        if email in users["users"]:
            return jsonify({"success": False, "error": "An account with that email already exists. Try signing in."}), 409
        # Keep anything already entered in this browser before the person made an account.
        owned = {u["cid"] for u in users["users"].values()}
        cid = request.headers.get("X-Client-Id", "")
        if not CLIENT_ID_RE.match(cid) or cid in owned:
            cid = "acct_" + secrets.token_hex(12)
        users["users"][email] = {"email": email, "name": name, "cid": cid,
                                 "hash": generate_password_hash(password), "created": time.time()}
        token = _new_token(users, email)
        _write_users(users)
    tz = request_tz()

    def setup(state):
        state["profile"].update({"name": name, "age": age, "timezone": state["profile"].get("timezone") or tz,
                                 "allergies": clean_text(data.get("allergies"), "", 500),
                                 "language": data.get("language") if data.get("language") in LANGUAGE_NAMES else "en"})
        state["caregiver"]["patient_name"] = name
    mutate(cid, setup)
    return signed_in_response({"success": True, "token": token, "name": name}, token)


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    if not login_allowed(client_ip()):
        return jsonify({"success": False, "error": "Too many attempts. Please wait a few minutes."}), 429
    data = request.get_json(silent=True) or {}
    email = clean_text(data.get("email"), "", 254).lower()
    password = str(data.get("password") or "")
    with USERS_LOCK:
        users = _read_users()
        user = users["users"].get(email)
        if not user:
            return jsonify({"success": False, "no_account": True,
                            "error": f"There's no VeraDose account for {email} on this server yet. "
                                     "Check the spelling, or choose Sign up to create one."}), 401
        if not check_password_hash(user["hash"], password):
            return jsonify({"success": False, "error": "That password isn't right. Tap the eye icon to see what's "
                                                      "typed, and check Caps Lock."}), 401
        token = _new_token(users, email)
        _write_users(users)
    return signed_in_response({"success": True, "token": token, "name": user["name"]}, token)


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    with USERS_LOCK:
        users = _read_users()
        changed = False
        for tok in request_tokens():
            changed = users["tokens"].pop(_token_key(tok), None) is not None or changed
        if changed:
            _write_users(users)
    return signed_out_response({"success": True})


@app.route("/api/auth/delete", methods=["POST"])
def auth_delete():
    """Permanently delete the signed-in account and everything saved for it."""
    if not login_allowed(client_ip()):
        return jsonify({"success": False, "error": "Too many attempts. Please wait a few minutes."}), 429
    account = account_for_request()
    if not account:
        return jsonify({"success": False, "error": "Please sign in to continue.", "auth": False}), 401
    password = str((request.get_json(silent=True) or {}).get("password") or "")
    if not check_password_hash(account["hash"], password):
        return jsonify({"success": False, "error": "That password isn't right, so your account was not deleted."}), 403
    with USERS_LOCK:
        users = _read_users()
        users["users"].pop(account["email"], None)
        users["tokens"] = {k: v for k, v in users["tokens"].items() if v.get("email") != account["email"]}
        _write_users(users)
    with DATA_LOCK:
        store = _read_store()
        if store.pop(account["cid"], None) is not None:
            _write_store(store)
    return signed_out_response({"success": True})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    account = account_for_request()
    if not account:
        return jsonify({"success": False, "error": "Not signed in.", "auth": False}), 401
    return jsonify({"success": True, "name": account["name"], "email": account["email"]})


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
    who = caregiver.get("patient_name") or "A VeraDose user"
    greet = f"Hi {caregiver['name']}, " if caregiver.get("name") else "Hi, "
    body = (f"{greet}{who} added you as their caregiver in VeraDose. You'll get a message right away "
            f"if they miss a dose or their pills are running low. No action is needed now.")
    deliver_alert(caregiver, "You're now a VeraDose caregiver", body)


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
    who = caregiver.get("patient_name") or "A VeraDose user"
    body = (f"EMERGENCY: {who} pressed the Emergency button in VeraDose. Please contact them right away. "
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

    who = caregiver.get("patient_name") or "A VeraDose user"
    body = (f"VeraDose test message: {who} listed you as a caregiver and will send you a note "
            f"if a critical medication isn't marked as taken. No action is needed.")
    delivered = deliver_alert(caregiver, "VeraDose test message", body)
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
            who = cg.get("patient_name") or state["profile"].get("name") or "A VeraDose user"
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
                            f"(Sent by VeraDose because you were listed as a caregiver.)")
                    pending.append(("dose", cid, med["id"], now.date().isoformat(), dict(cg),
                                    f"VeraDose: missed dose of {med['name']}", body))

            for med in state["meds"]:
                key = str(med["id"])
                if is_low(state, med):
                    if med["id"] in state["low_alerted"] or int(state["low_tries"].get(key, 0)) >= 3:
                        continue
                    left = days_left(med)
                    body = (f"{greet}{who} is running low on {med['name']}: {med['stock']} pill(s) left, "
                            f"about {left:.0f} day(s) of doses. Please help them arrange a refill. "
                            f"(Sent by VeraDose because you were listed as a caregiver.)")
                    pending.append(("low", cid, med["id"], None, dict(cg),
                                    f"VeraDose: {med['name']} is running low", body))
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

if os.environ.get("RENDER") and not DATABASE_URL and not DATABASE_PROBLEM:
    print("\n" + "!" * 78 + "\n[VeraDose] WARNING: no DATABASE_URL is set. Render's free plan erases saved files on every"
          "\nrestart, so ALL ACCOUNTS AND MEDICINES WILL BE LOST. Add a free Neon database and set DATABASE_URL in Render's Environment tab.\n" + "!" * 78 + "\n")
elif DATABASE_URL:
    print("[VeraDose] Saving accounts and medicines in the database (DATABASE_URL). Checking the connection...")
    threading.Thread(target=_check_database_at_startup, name="db-check", daemon=True).start()


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
        "You are Buddy, VeraDose's friendly medication helper. Many users are older adults, so keep every reply "
        "calm, warm and easy to read. "
        "STYLE RULES: Keep answers under about 90 words. Start with one short, kind sentence that answers the question. "
        "Then give at most 3 short bullet points, each one simple sentence. Use plain everyday words, not medical jargon "
        "or numbers like blood-pressure readings unless the user asks. No headings, no tables, and no more than one list. "
        "You may make one or two key words **bold**. End with one short line about who to contact if needed. "
        "If the user wants more detail, they can ask, so don't try to cover everything at once. "
        "SAFETY: You are not a doctor: never change doses or tell the user to stop a prescription; for those, tell them "
        "to talk to their pharmacist or doctor. If the user describes something that could be an emergency (chest pain, "
        "trouble breathing, fainting, confusion, signs of stroke, a severe allergic reaction, or a possible overdose), "
        "your FIRST sentence must tell them to call 911 now or press the Emergency button. "
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
@require_client
def analyze(cid):
    """Scan a prescription label and/or medicine bottle (up to 3 photos) and return form fields.
    Nothing is saved here: the person checks the filled-in form and presses Save themselves."""
    uploads = [f for f in request.files.getlist("file") if f and f.filename][:3]
    if not uploads:
        return jsonify({"success": False, "error": "No prescription or bottle photo received."}), 400
    if not client:
        return jsonify({"success": False, "error": "Label scanning needs the AI key (GROQ_API_KEY) on the server."}), 503
    images = []
    for upload in uploads:
        try:
            images.append(encode_image(Image.open(io.BytesIO(upload.read()))))
        except Exception:
            return jsonify({"success": False, "error": "One of the photos could not be read as an image."}), 400

    prompt = (
        "These photos show a prescription label and/or a medicine bottle for ONE medication. Read them together. "
        "Report ONLY what is actually printed. Never guess or invent a value; use \"\" or 0 when it isn't shown. "
        "Reply with ONLY a JSON object with these keys:\n"
        '"name": medication name as printed (e.g. "Metformin HCl"),\n'
        '"dosage": strength (e.g. "500 mg"),\n'
        '"pills_per_dose": number of tablets/capsules per dose as an integer (e.g. "take 2 tablets" -> 2), or 0,\n'
        '"times_per_day": how many times a day (once daily -> 1, twice daily -> 2, every 8 hours -> 3), or 0,\n'
        '"time": a specific time if printed, as 24-hour "HH:MM", else "". Use "08:00" for "in the morning", '
        '"21:00" for "at bedtime", "18:00" for "with dinner/evening", else "",\n'
        '"quantity": number of pills dispensed (e.g. "Qty: 60" -> 60), or 0,\n'
        '"directions": the directions exactly as printed, short,\n'
        '"food_guide": any food or timing advice printed (e.g. "Take with food"), short, or "",\n'
        '"badges": a list using ONLY these exact words that the label clearly states: '
        '"with_food","empty_stomach","avoid_dairy","avoid_alcohol","avoid_sugar","before_bed","with_water",'
        '"before_lunch","before_breakfast","before_dinner",\n'
        '"shape": pill shape if the label describes it (round, oval, oblong, square, diamond, triangle, hexagon), else "",\n'
        '"color": pill color if described (white, yellow, pink, red, purple, green, blue, brown, orange), else "",\n'
        '"imprint": imprint letters/numbers if described, else "",\n'
        '"readable": true if a medication label was visible, false if not.'
    )
    content = [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]
    try:
        raw = ai_complete([{"role": "user", "content": content}], vision=True, temperature=0.1, max_tokens=900)
    except Exception as exc:
        return jsonify({"success": False, "error": ai_error_message(exc)}), 502

    data = extract_json(raw)
    if not data or data.get("readable") is False or not clean_text(data.get("name"), "", 80):
        return jsonify({"success": False, "error": "VeraDose couldn't read a medication label in that photo. "
                        "Try again closer, in good light, with the label flat and in focus."}), 422
    time_str = str(data.get("time") or "").strip()
    pill = clean_pill(data)
    parsed = {
        "name": clean_text(data.get("name"), "", 80),
        "dosage": clean_text(data.get("dosage"), "", 60),
        "pills_per_dose": clamp_int(data.get("pills_per_dose"), 0, 0, 20),
        "times_per_day": clamp_int(data.get("times_per_day"), 0, 0, 12),
        "time": time_str if TIME_RE.match(time_str) else "",
        "quantity": clamp_int(data.get("quantity"), 0, 0, 10000),
        "directions": clean_text(data.get("directions"), "", 300),
        "food_guide": clean_text(data.get("food_guide"), "", 200),
        "badges": [b for b in dict.fromkeys(data.get("badges") or []) if b in VALID_BADGES and b != "avoid_grapefruit"],
        "shape": pill["shape"] if pill["shape"] != "capsule" else "",
        "color": pill["color"] if pill["color"] != "gray" else "",
        "imprint": pill["imprint"],
    }
    return jsonify({"success": True, "parsed": parsed, "photos": len(images)})


# ==========================================================================
# Error handlers: always return something readable, never a blank page
# ==========================================================================
@app.errorhandler(StorageBusy)
def handle_storage_busy(_err):
    if DATABASE_URL:
        return jsonify({"success": False, "error": "VeraDose can't reach its database right now. Nothing was lost. "
                                                   "Please try again in a minute."}), 503
    return jsonify({"success": False, "error": "VeraDose couldn't open its saved data for a moment. "
                                               "Nothing was lost. Please try again."}), 503


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
    return Response("<h1>Page not found</h1><p><a href='/'>Back to VeraDose</a></p>", status=404, mimetype="text/html")


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
    print(f"\nVeraDose running at http://127.0.0.1:{port}")
    try:
        n = len(_read_users()["users"])
        where = "the database (DATABASE_URL)" if DATABASE_URL else USERS_FILE
        print(f"Accounts are saved in: {where} ({n} account{'s' if n != 1 else ''})\n")
    except Exception as exc:
        print(f"[warn] Could not read the accounts file yet: {exc}\n")
    if not os.environ.get("RENDER"):
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass
    try:
        app.run(host=host, port=port, debug=False)
    except OSError as exc:
        print(f"\n[VeraDose] Could not start on port {port}: {exc}")
        print("  - Is VeraDose already running in another terminal? Close that one (Ctrl+C) and try again.")
        print("  - On a Mac, turn off System Settings > General > AirDrop & Handoff > AirPlay Receiver (it uses port 5000).")
        print("  - Or start VeraDose on another port, e.g.  PORT=5050 python app.py\n")