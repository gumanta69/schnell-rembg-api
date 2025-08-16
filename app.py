import os
import io
import base64
import logging
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import requests

from rembg import remove
import rembg as rembg_pkg

app = Flask(__name__, static_folder="static")
CORS(app)

HF_API_KEY = os.getenv("HF_API_KEY", "")
SCHNELL_MODEL_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
PORT = int(os.getenv("PORT", "5000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("schnell-rembg-api")

def hf_generate_image(prompt: str) -> io.BytesIO | None:
    if not HF_API_KEY:
        logger.error("HF_API_KEY not set")
        return None
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"inputs": prompt}
    r = requests.post(SCHNELL_MODEL_URL, headers=headers, json=data, timeout=180)
    if r.status_code != 200:
        logger.error("HF API error %s: %s", r.status_code, r.text[:500])
        return None
    return io.BytesIO(r.content)

def rembg_bytes(img_bytes: bytes) -> io.BytesIO:
    out = remove(img_bytes)
    return io.BytesIO(out)

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/rembg-version", methods=["GET"])
def rembg_version():
    return jsonify({"version": rembg_pkg.__version__})

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    buf = hf_generate_image(prompt)
    if buf is None:
        return jsonify({"error": "Image generation failed (HF API)."}), 502
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/rembg", methods=["POST"])
def rembg_file():
    if "image" not in request.files:
        return jsonify({"error": "No image file. Use field name 'image'."}), 400
    raw = request.files["image"].read()
    if not raw:
        return jsonify({"error": "Empty file"}), 400
    try:
        out = rembg_bytes(raw)
        out.seek(0)
        return send_file(out, mimetype="image/png")
    except Exception as e:
        logger.exception("rembg error")
        return jsonify({"error": str(e)}), 500

@app.route("/rembg-b64", methods=["POST"])
def rembg_b64():
    data = request.get_json(silent=True) or {}
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image (base64) in JSON"}), 400
    if "," in img_b64:
        img_b64 = img_b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(img_b64)
        out = rembg_bytes(raw)
        out.seek(0)
        return send_file(out, mimetype="image/png")
    except Exception as e:
        logger.exception("rembg-b64 error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
