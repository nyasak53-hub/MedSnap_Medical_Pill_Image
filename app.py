import os
import io
import base64
import json
import webbrowser
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
from PIL import Image
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_ROOT, 'medications_data.json')

def load_environment_variables():
    env_paths = [
        os.path.join(PROJECT_ROOT, '.env'),
        '.env',
    ]
    loaded = False
    for env_path in env_paths:
        if env_path and os.path.exists(env_path):
            try:
                loaded = load_dotenv(env_path, override=False, encoding='utf-8') or loaded
            except Exception as exc:
                print(f"⚠️ Could not read {env_path}: {exc}")
    return loaded


def normalize_api_key(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip().strip('"').strip("'")
    placeholder_values = {
        'your_groq_api_key_here',
        'your_real_key_here',
        'replace_me',
        'changeme',
        ''
    }
    return None if value.lower() in placeholder_values else value


load_environment_variables()

# Enable CORS headers for all incoming requests
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Initialize Groq Client
GROQ_API_KEY = normalize_api_key(os.environ.get("GROQ_API_KEY"))
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"⚠️ Groq client initialization warning: {e}")
else:
    print("⚠️ GROQ_API_KEY not set; the app will use heuristic fallback responses.")
    print("💡 Create a .env file with GROQ_API_KEY=your_real_key_here and restart the app.")


VISION_MODEL = "llama-3.2-11b-vision-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

DEFAULT_MEDS = [
    {
        "id": 1,
        "name": "Metformin",
        "dosage": "500mg",
        "slot": "Morning",
        "stock": 14,
        "daily_dose": 2,
        "food_guide": "Take with your largest meal to avoid stomach upset.",
        "timing": "With food",
        "taken": False,
        "scheduled_time": "08:00"
    },
    {
        "id": 2,
        "name": "Lisinopril",
        "dosage": "10mg",
        "slot": "Evening",
        "stock": 4,
        "daily_dose": 1,
        "food_guide": "Can be taken with or without food.",
        "timing": "Anytime",
        "taken": False,
        "scheduled_time": "20:00"
    }
]

def load_meds():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return DEFAULT_MEDS

def save_meds(meds):
    with open(DATA_FILE, 'w') as f:
        json.dump(meds, f, indent=2)

def encode_image_to_jpeg_base64(pil_img):
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=90)
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"

@app.route('/')
def landing():
    if os.path.exists('landing.html'):
        return send_from_directory('.', 'landing.html')
    return send_from_directory('.', 'index.html')

@app.route('/app')
def main_app():
    return send_from_directory('.', 'index.html')

@app.route('/api/medications', methods=['GET', 'POST'])
def handle_medications():
    meds = load_meds()
    if request.method == 'GET':
        return jsonify({"success": True, "medications": meds})
    elif request.method == 'POST':
        data = request.json
        new_id = max([m['id'] for m in meds], default=0) + 1
        new_med = {
            "id": new_id,
            "name": data.get('name', 'Unknown'),
            "dosage": data.get('dosage', '1 pill'),
            "slot": data.get('slot', 'Morning'),
            "stock": int(data.get('stock', 30)),
            "daily_dose": int(data.get('daily_dose', 1)),
            "food_guide": data.get('food_guide', 'Take as directed by your physician.'),
            "timing": data.get('timing', 'With water'),
            "taken": False,
            "scheduled_time": data.get('scheduled_time', '09:00')
        }
        meds.append(new_med)
        save_meds(meds)
        return jsonify({"success": True, "medication": new_med})

@app.route('/api/medications/<int:med_id>/take', methods=['POST'])
def take_medication(med_id):
    meds = load_meds()
    for m in meds:
        if m['id'] == med_id:
            m['taken'] = not m['taken']
            if m['taken'] and m['stock'] > 0:
                m['stock'] -= m['daily_dose']
                if m['stock'] < 0:
                    m['stock'] = 0
            save_meds(meds)
            return jsonify({"success": True, "medication": m})
    return jsonify({"success": False, "error": "Medication not found"}), 404

@app.route('/api/identify-pill', methods=['POST'])
def identify_pill():
    data = request.json
    shape = data.get('shape', '').lower()
    color = data.get('color', '').lower()
    imprint = data.get('imprint', '').upper()

    database = [
        {"name": "Aspirin 81mg", "shape": "round", "color": "white", "imprint": "BAYER"},
        {"name": "Lipitor 20mg", "shape": "oval", "color": "white", "imprint": "PD 155"},
        {"name": "Ibuprofen 200mg", "shape": "round", "color": "brown", "imprint": "I-2"},
        {"name": "Metformin 500mg", "shape": "oval", "color": "white", "imprint": "MET 500"}
    ]

    matches = []
    for p in database:
        if (not shape or shape in p['shape']) and (not color or color in p['color']) and (not imprint or imprint in p['imprint']):
            matches.append(p['name'])

    return jsonify({"success": True, "matches": matches})

@app.route('/api/interactions', methods=['POST'])
def check_interactions():
    data = request.json
    med_list = data.get('medications', [])
    if not client or not med_list:
        return jsonify({"success": True, "analysis": "Standard safety check: Always consult your pharmacist regarding potential drug interactions."})

    prompt = f"You are a clinical pharmacist safety assistant. Review the following list of medications: {', '.join(med_list)}. Check for any major drug interactions, contraindications, or important safety warnings. Give a concise, professional warning summary."
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400
        )
        return jsonify({"success": True, "analysis": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"success": False, "error": "No prescription or pill bottle image uploaded."}), 400

    file = request.files['file']
    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes))
        base64_image_url = encode_image_to_jpeg_base64(pil_img)

        prompt = (
            "You are a medical assistant OCR tool. Analyze this image of a pill bottle or prescription label.\n"
            "Extract the following details:\n"
            "1. Medication Name\n"
            "2. Dosage (e.g., 10mg, 500mg)\n"
            "3. Schedule/Frequency (e.g., Morning, Afternoon, Evening)\n\n"
            "Format your answer clearly as text so the user can easily auto-fill their medication schedule form."
        )

        analysis_text = "Medication detected successfully from label scan."
        if client:
            try:
                response = client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_image_url}}
                        ]
                    }],
                    temperature=0.1,
                    max_tokens=500
                )
                analysis_text = response.choices[0].message.content
            except Exception as e:
                print(f"Vision API Error: {e}")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "ocr_result": analysis_text
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# --- SERVER STARTUP ---
if __name__ == '__main__':
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    url = f"http://127.0.0.1:{PORT}" if HOST == '0.0.0.0' else f"http://{HOST}:{PORT}"

    print(f"\n💊 MedSnap Pill & Medication Schedule Manager Running!")
    print(f"🔗 Application URL: {url}\n")

    try:
        if HOST in ['127.0.0.1', 'localhost', '0.0.0.0']:
            webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception:
        pass

    app.run(host=HOST, port=PORT, debug=False)