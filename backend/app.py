"""
app.py — API-only backend
--------------------------
Same pipeline as the original Flask app, but instead of rendering HTML
templates, it exposes a JSON API that a decoupled frontend (hosted on
Vercel) calls over HTTPS.

Routes:
  GET  /              -> health check (used by HF Spaces / Cloud Run)
  POST /predict        -> multipart/form-data { image: <file> }
                          returns JSON with predictions, clinical info,
                          and both Grad-CAM overlays as base64 PNGs.

Listens on $PORT (defaults to 7860, the Hugging Face Spaces default).
"""

import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from utils.predict import load_and_preprocess_image_from_bytes, predict_top_k, get_model
from utils.clinical import get_clinical_info, DISCLAIMER
from utils.gradcam import generate_gradcam_images

app = Flask(__name__)

# Allow requests from your Vercel frontend. Restrict this to your actual
# domain(s) once deployed, e.g.:
# CORS(app, origins=["https://your-frontend.vercel.app"])
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img_rgb_uint8):
    """Encode an RGB uint8 numpy array as a base64 PNG data string."""
    success, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode image")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Skin disease prediction API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part in the request. Expected field name 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a PNG or JPG image."}), 400

    try:
        image_bytes = file.read()

        # 1. Preprocess + predict
        img_batch_0_255, img_0_1 = load_and_preprocess_image_from_bytes(image_bytes)
        top_results, preds = predict_top_k(img_batch_0_255, top_k=3)
        top_prediction = top_results[0]

        # 2. Clinical info for the top prediction
        clinical_info = get_clinical_info(top_prediction["label"])

        # 3. Grad-CAM before / after MSAM
        model = get_model()
        pred_index = int(preds.argmax())
        overlay_before, overlay_after = generate_gradcam_images(
            model, img_batch_0_255, img_0_1, pred_index=pred_index
        )

        response = {
            "top_results": top_results,
            "top_prediction": top_prediction,
            "clinical_info": clinical_info,
            "disclaimer": DISCLAIMER,
            "gradcam_before": image_to_base64(overlay_before),
            "gradcam_after": image_to_base64(overlay_after),
        }

        return jsonify(response)

    except Exception as e:
        # Don't leak stack traces to the client in production; log server-side instead.
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
