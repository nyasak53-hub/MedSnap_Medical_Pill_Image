import os
import io
import re
import json
import base64
import tempfile
import threading
import webbrowser
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT: static_folder=None. The original code served the whole project
# folder as static files, which would have exposed app.py, your data file and
# anything else in the folder to anyone on the internet. Pages are now served
# only through the explicit routes below.
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
CORS(app, resources={r"/api/*": {"origins": "*"}})

load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)

# On Render, point DATA_DIR at a persistent disk (e.g. /var/data) if you add one.
# Without a disk, saved medications reset whenever the service redeploys/restarts.
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)
DATA_FILE = os.path.join(DATA_DIR, "medications_data.json")
DATA_LOCK = threading.Lock()


def normalize_api_key(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip().strip('"').strip("'")
    placeholders = {"your_groq_api_key_here", "your_real_key_here", "replace_me", "changeme", ""}
    return None if value.lower() in placeholders else value


GROQ_API_KEY = normalize_api_key(os.environ.get("GROQ_API_KEY"))
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as exc:
        print(f"[warn] Groq client initialization failed: {exc}")
else:
    print("[warn] GROQ_API_KEY not set; AI features will be disabled.")

# Models can be changed from Render's Environment tab without touching code.
# Vision models are tried in order until one works.
VISION_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GROQ_VISION_MODELS",
        "meta-llama/llama-4-scout-17b-16e-instruct,qwen/qwen3.6-27b",
    ).split(",")
    if m.strip()
]
TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

VALID_SLOTS = ("Morning", "Afternoon", "Evening")

DEFAULT_MEDS = [
    {
        "id": 1, "name": "Metformin", "dosage": "500mg", "slot": "Morning",
        "stock": 14, "daily_dose": 2,
        "food_guide": "Take with your largest meal to avoid stomach upset.",
        "timing": "With food", "taken": False, "scheduled_time": "08:00",
    },
    {
        "id": 2, "name": "Lisinopril", "dosage": "10mg", "slot": "Evening",
        "stock": 4, "daily_dose": 1,
        "food_guide": "Can be taken with or without food.",
        "timing": "Anytime", "taken": False, "scheduled_time": "20:00",
    },
]

# --------------------------------------------------------------------------
# Storage: one list of medications per browser (X-Client-Id header), so
# visitors never see each other's medication lists.
# --------------------------------------------------------------------------
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


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
            json.dump(store, f, indent=2)
        os.replace(tmp_path, DATA_FILE)  # atomic: never leaves a half-written file
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_meds(client_id):
    with DATA_LOCK:
        store = _read_store()
        if client_id not in store:
            store[client_id] = json.loads(json.dumps(DEFAULT_MEDS))
            try:
                _write_store(store)
            except Exception as exc:
                print(f"[warn] Could not save data: {exc}")
        return store[client_id]


def save_meds(client_id, meds):
    with DATA_LOCK:
        store = _read_store()
        store[client_id] = meds
        _write_store(store)


def require_client(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        cid = request.headers.get("X-Client-Id", "")
        if not CLIENT_ID_RE.match(cid):
            return jsonify({"success": False, "error": "Missing or invalid client id."}), 400
        return fn(cid, *args, **kwargs)
    return wrapper


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


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_response(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.isfile(path):
        return send_from_directory(BASE_DIR, filename)
    return None


@app.route("/")
def landing():
    return (
        page_response("landing.html")
        or page_response("index.html")
        or Response(
            "<h1>MedSnap is running</h1><p>The server started, but <code>index.html</code> "
            "was not found next to <code>app.py</code> in your repository. Make sure it is "
            "committed and pushed to GitHub.</p>",
            status=500,
            mimetype="text/html",
        )
    )


@app.route("/app")
def main_app():
    return page_response("index.html") or landing()


@app.route("/healthz")
def healthz():
    """Used by Render's health check and handy for debugging."""
    return jsonify({
        "status": "ok",
        "index_html_found": os.path.isfile(os.path.join(BASE_DIR, "index.html")),
        "ai_enabled": client is not None,
    })


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.route("/api/medications", methods=["GET", "POST"])
@require_client
def handle_medications(client_id):
    meds = load_meds(client_id)

    if request.method == "GET":
        return jsonify({"success": True, "medications": meds})

    data = request.get_json(silent=True) or {}
    name = clean_text(data.get("name"))
    if not name:
        return jsonify({"success": False, "error": "Medication name is required."}), 400
    if len(meds) >= 100:
        return jsonify({"success": False, "error": "Medication limit reached."}), 400

    new_med = {
        "id": max([m["id"] for m in meds], default=0) + 1,
        "name": name,
        "dosage": clean_text(data.get("dosage"), "1 pill", 60),
        "slot": normalize_slot(data.get("slot")),
        "stock": clamp_int(data.get("stock"), 30, 0, 10000),
        "daily_dose": clamp_int(data.get("daily_dose"), 1, 1, 20),
        "food_guide": clean_text(data.get("food_guide"), "Take as directed by your physician.", 200),
        "timing": clean_text(data.get("timing"), "With water", 60),
        "taken": False,
        "scheduled_time": clean_text(data.get("scheduled_time"), "09:00", 5),
    }
    meds.append(new_med)
    save_meds(client_id, meds)
    return jsonify({"success": True, "medication": new_med})


@app.route("/api/medications/<int:med_id>", methods=["DELETE"])
@require_client
def delete_medication(client_id, med_id):
    meds = load_meds(client_id)
    remaining = [m for m in meds if m["id"] != med_id]
    if len(remaining) == len(meds):
        return jsonify({"success": False, "error": "Medication not found"}), 404
    save_meds(client_id, remaining)
    return jsonify({"success": True})


@app.route("/api/medications/<int:med_id>/take", methods=["POST"])
@require_client
def take_medication(client_id, med_id):
    meds = load_meds(client_id)
    for m in meds:
        if m["id"] == med_id:
            m["taken"] = not m["taken"]
            if m["taken"]:
                m["stock"] = max(0, m["stock"] - m["daily_dose"])
            else:
                m["stock"] += m["daily_dose"]  # undo gives the pills back
            save_meds(client_id, meds)
            return jsonify({"success": True, "medication": m})
    return jsonify({"success": False, "error": "Medication not found"}), 404


@app.route("/api/identify-pill", methods=["POST"])
def identify_pill():
    data = request.get_json(silent=True) or {}
    shape = str(data.get("shape", "")).lower().strip()
    color = str(data.get("color", "")).lower().strip()
    imprint = str(data.get("imprint", "")).upper().strip()

    database = [
        {"name": "Aspirin 81mg", "shape": "round", "color": "white", "imprint": "BAYER"},
        {"name": "Lipitor 20mg", "shape": "oval", "color": "white", "imprint": "PD 155"},
        {"name": "Ibuprofen 200mg", "shape": "round", "color": "brown", "imprint": "I-2"},
        {"name": "Metformin 500mg", "shape": "oval", "color": "white", "imprint": "MET 500"},
    ]

    matches = [
        p["name"] for p in database
        if (not shape or shape == p["shape"])
        and (not color or color == p["color"])
        and (not imprint or imprint in p["imprint"])
    ]
    return jsonify({"success": True, "matches": matches})


@app.route("/api/interactions", methods=["POST"])
def check_interactions():
    data = request.get_json(silent=True) or {}
    med_list = [clean_text(m, "", 80) for m in (data.get("medications") or [])][:30]
    med_list = [m for m in med_list if m]

    if not med_list:
        return jsonify({"success": False, "error": "Add at least one medication first."}), 400
    if not client:
        return jsonify({
            "success": True,
            "analysis": "AI checking is not configured on this server. Please ask your "
                        "pharmacist about potential drug interactions.",
        })

    prompt = (
        "You are a clinical pharmacist safety assistant. Review this list of medications: "
        f"{', '.join(med_list)}. Check for major drug interactions, contraindications, and "
        "important safety warnings. Give a concise, professional summary in plain text. "
        "End with a reminder to confirm with a pharmacist or doctor."
    )
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return jsonify({"success": True, "analysis": response.choices[0].message.content})
    except Exception as exc:
        print(f"[error] interactions: {exc}")
        return jsonify({"success": False, "error": "The AI service could not be reached. Try again shortly."}), 502


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
    pil_img.thumbnail((1600, 1600))  # smaller upload = faster + cheaper
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
        pil_img = Image.open(io.BytesIO(upload.read()))
        image_url = encode_image(pil_img)
    except Exception:
        return jsonify({"success": False, "error": "That file could not be read as an image."}), 400

    prompt = (
        "You are an OCR tool reading a pill bottle or prescription label. "
        "Reply with ONLY a JSON object, no other text, with these keys: "
        '"name" (medication name), "dosage" (e.g. "10mg"), '
        '"slot" (one of "Morning", "Afternoon", "Evening", based on the directions), '
        '"directions" (the label directions, short). '
        'Use an empty string for anything you cannot read.'
    )

    last_error = None
    for model in VISION_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
                temperature=0.1,
                max_tokens=500,
            )
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
            last_error = exc
            print(f"[error] vision model {model} failed: {exc}")

    return jsonify({
        "success": False,
        "error": "The AI vision service failed. Check the Render logs and the GROQ_VISION_MODELS setting.",
    }), 502


# --------------------------------------------------------------------------
# Error handlers: always return something readable, never a blank page
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Local development only (Render uses gunicorn and never runs this block)
# --------------------------------------------------------------------------
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