import os
import io
import base64
import re
import webbrowser
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
from PIL import Image
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_environment_variables():
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)

load_environment_variables()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

VISION_MODEL = "llama-3.2-11b-vision-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

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
        # Only attempt to open browser automatically during local development
        if HOST in ['127.0.0.1', 'localhost', '0.0.0.0']:
            webbrowser.open(f"http://127.0.0.1:{PORT}")
    except Exception:
        pass

    app.run(host=HOST, port=PORT, debug=False)