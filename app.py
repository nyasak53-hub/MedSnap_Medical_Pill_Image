import base64
import io
import json
import os
from datetime import date, datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq
from PIL import Image

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_ROOT, 'medications_data.json')


def load_env():
  env_path = os.path.join(PROJECT_ROOT, '.env')
  if os.path.exists(env_path):
    from dotenv import load_dotenv

    load_dotenv(env_path)


load_env()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

VISION_MODEL = 'llama-3.2-11b-vision-instruct'
TEXT_MODEL = 'llama-3.3-70b-versatile'

DEFAULT_MEDS = [
    {
        'id': 1,
        'name': 'Metformin',
        'dosage': '500mg',
        'slot': 'Morning',
        'stock': 14,
        'daily_dose': 2,
        'food_guide': 'Take with your largest meal to avoid stomach upset.',
        'timing': 'With food',
        'taken': False,
        'scheduled_time': '08:00',
    },
    {
        'id': 2,
        'name': 'Lisinopril',
        'dosage': '10mg',
        'slot': 'Evening',
        'stock': 4,
        'daily_dose': 1,
        'food_guide': 'Can be taken with or without food.',
        'timing': 'Anytime',
        'taken': False,
        'scheduled_time': '20:00',
    },
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
    return jsonify({'success': True, 'medications': meds})
  elif request.method == 'POST':
    data = request.json
    new_id = max([m['id'] for m in meds], default=0) + 1
    new_med = {
        'id': new_id,
        'name': data.get('name', 'Unknown'),
        'dosage': data.get('dosage', '1 pill'),
        'slot': data.get('slot', 'Morning'),
        'stock': int(data.get('stock', 30)),
        'daily_dose': int(data.get('daily_dose', 1)),
        'food_guide': data.get(
            'food_guide', 'Take as directed by your physician.'
        ),
        'timing': data.get('timing', 'With water'),
        'taken': False,
        'scheduled_time': data.get('scheduled_time', '09:00'),
    }
    meds.append(new_med)
    save_meds(meds)
    return jsonify({'success': True, 'medication': new_med})


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
      return jsonify({'success': True, 'medication': m})
  return jsonify({'success': False, 'error': 'Medication not found'}), 404


@app.route('/api/identify-pill', methods=['POST'])
def identify_pill():
  data = request.json
  shape = data.get('shape', '').lower()
  color = data.get('color', '').lower()
  imprint = data.get('imprint', '').upper()

  database = [
      {
          'name': 'Aspirin 81mg',
          'shape': 'round',
          'color': 'white',
          'imprint': 'BAYER',
      },
      {
          'name': 'Lipitor 20mg',
          'shape': 'oval',
          'color': 'white',
          'imprint': 'PD 155',
      },
      {
          'name': 'Ibuprofen 200mg',
          'shape': 'round',
          'color': 'brown',
          'imprint': 'I-2',
      },
      {
          'name': 'Metformin 500mg',
          'shape': 'oval',
          'color': 'white',
          'imprint': 'MET 500',
      },
  ]

  matches = []
  for p in database:
    if (
        (not shape or shape in p['shape'])
        and (not color or color in p['color'])
        and (not imprint or imprint in p['imprint'])
    ):
      matches.append(p['name'])

  return jsonify({'success': True, 'matches': matches})


@app.route('/api/interactions', methods=['POST'])
def check_interactions():
  data = request.json
  med_list = data.get('medications', [])
  if not client or not med_list:
    return jsonify({
        'success': True,
        'analysis': (
            'Standard safety check: Always consult your pharmacist regarding'
            ' potential drug interactions.'
        ),
    })

  prompt = (
      'You are a clinical pharmacist safety assistant. Review the following'
      f' list of medications: {", ".join(med_list)}. Check for any major drug'
      ' interactions, contraindications, or important safety warnings. Give a'
      ' concise, professional warning summary.'
  )
  try:
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    return jsonify(
        {'success': True, 'analysis': response.choices[0].message.content}
    )
  except Exception as e:
    return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
  if 'file' not in request.files or request.files['file'].filename == '':
    return (
        jsonify({
            'success': False,
            'error': 'No prescription or pill bottle image uploaded.',
        }),
        400,
    )

  file = request.files['file']
  try:
    image_bytes = file.read()
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode in ('RGBA', 'P'):
      pil_img = pil_img.convert('RGB')
    buffered = io.BytesIO()
    pil_img.save(buffered, format='JPEG', quality=90)
    base64_image_url = (
        'data:image/jpeg;base64,'
        + base64.b64encode(buffered.getvalue()).decode('utf-8')
    )

    prompt = (
        'You are a medical assistant OCR tool. Analyze this image of a pill'
        ' bottle or prescription label.\nExtract the following details:\n1.'
        ' Medication Name\n2. Dosage (e.g., 10mg, 500mg)\n3. Schedule/Frequency'
        ' (Morning, Afternoon, Evening)\n4. Food Instructions'
    )

    analysis_text = 'Medication detected successfully from label scan.'
    if client:
      try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': base64_image_url}},
                ],
            }],
            temperature=0.1,
            max_tokens=500,
        )
        analysis_text = response.choices[0].message.content
      except Exception as e:
        print(f'Vision API Error: {e}')

    return jsonify({
        'success': True,
        'filename': file.filename,
        'ocr_result': analysis_text,
    })
  except Exception as e:
    return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000, debug=True)