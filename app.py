"""
app.py
------
Flask web app for the AI-Assisted Skin Disease Prediction system.

Pipeline per request:
  1. User uploads a skin image via the index page.
  2. Image is saved to uploads/.
  3. utils.predict loads (cached) the EfficientNetV2-S + MSAM model and
     returns the top-3 predicted classes with confidence scores.
  4. utils.clinical looks up description / symptoms / causes / treatment /
     advice for the top predicted class.
  5. utils.gradcam generates Grad-CAM heatmap overlays BEFORE and AFTER the
     MSAM attention block, saved to static/ for display.
  6. result.html renders everything.
"""

import os
import uuid

import cv2
from flask import Flask, render_template, request, redirect, url_for, flash

from utils.predict import load_and_preprocess_image, predict_top_k, get_model
from utils.clinical import get_clinical_info, DISCLAIMER
from utils.gradcam import generate_gradcam_images

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
app.secret_key = "dev-secret-key"  # change this in production

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG or JPG image.")
        return redirect(url_for("index"))

    # Save the uploaded image with a unique name
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_id = uuid.uuid4().hex[:8]
    saved_filename = f"upload_{unique_id}.{ext}"
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
    file.save(saved_path)

    # 1. Preprocess + predict
    img_batch_0_255, img_0_1 = load_and_preprocess_image(saved_path)
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

    gradcam_before_name = f"gradcam_before_{unique_id}.png"
    gradcam_after_name = f"gradcam_after_{unique_id}.png"

    cv2.imwrite(
        os.path.join(STATIC_FOLDER, gradcam_before_name),
        cv2.cvtColor(overlay_before, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(STATIC_FOLDER, gradcam_after_name),
        cv2.cvtColor(overlay_after, cv2.COLOR_RGB2BGR)
    )

    return render_template(
        "result.html",
        uploaded_image=saved_filename,
        top_results=top_results,
        top_prediction=top_prediction,
        clinical_info=clinical_info,
        disclaimer=DISCLAIMER,
        gradcam_before=gradcam_before_name,
        gradcam_after=gradcam_after_name,
    )


# Serve uploaded images (uploads/ isn't inside static/ by default)
from flask import send_from_directory


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
