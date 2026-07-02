# AI-Assisted Skin Disease Prediction

A Flask web app that classifies skin conditions from an uploaded photo using
an **EfficientNetV2-S backbone with a Multi-Scale Attention Module (MSAM)**,
trained on the DermNet dataset (23 classes). The app also shows:

- Top-3 predicted conditions with confidence scores
- Clinical information (description, symptoms, causes, treatment, advice)
  for the top prediction
- Grad-CAM heatmaps **before** and **after** the MSAM attention block, so you
  can visually compare what the backbone focuses on vs. what MSAM refines it to

This project was generated to accompany the `EfficientNetV2_MSAM.ipynb`
training notebook (backbone, MSAM/ECA layers, class-imbalance-aware
fine-tuning, and evaluation all happen there; this Flask app is the
inference/demo layer).

## Project structure

```
AI-Assisted-Skin-Disease-Prediction/
│
├── app.py                     # Flask entry point
├── requirements.txt
├── README.md
│
├── model/
│      best_balanced.keras     # trained model (you must copy this in — see below)
│
├── clinical_module/
│      clinical_info.json      # description/symptoms/causes/treatment per class
│
├── uploads/                   # uploaded images are saved here at runtime
│
├── static/
│      style.css
│      gradcam.png             # (generated per-request as gradcam_before/after_*.png)
│
├── templates/
│      index.html
│      result.html
│
├── utils/
│      predict.py              # model loading + inference
│      gradcam.py              # MSAM/ECABlock layer defs + Grad-CAM logic
│      clinical.py             # clinical_info.json lookup
│
├── sample_images/              # put a few test images here if you want
└── venv/                       # your local virtual environment (not included)
```

## 1. Get the trained model file

The notebook saves the best balanced model to Google Drive as:

```
/content/drive/MyDrive/Dermnet/MSAM_Balanced/best_balanced.keras
```

Download that file and place it at:

```
AI-Assisted-Skin-Disease-Prediction/model/best_balanced.keras
```

The app will refuse to start prediction (with a clear error) if this file is
missing.

## 2. Set up the environment

```bash
cd AI-Assisted-Skin-Disease-Prediction
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, upload a skin image,
and click **Analyze Image**.

## How it works

1. **Upload** — `templates/index.html` posts an image to `/predict`.
2. **Preprocess** — `utils/predict.py` resizes the image to 224×224 and keeps
   pixel values in the raw 0–255 range (EfficientNetV2 handles its own
   internal rescaling, matching how the model was trained — no manual
   normalization is applied).
3. **Inference** — the cached Keras model (loaded once, with `ECABlock` and
   `MSAM` registered via `custom_objects`) predicts a probability per class;
   the top 3 are shown with confidence bars.
4. **Clinical lookup** — `utils/clinical.py` reads
   `clinical_module/clinical_info.json` and returns the description,
   symptoms, causes, treatment, and advice for the top predicted class.
5. **Grad-CAM** — `utils/gradcam.py` builds two sub-models: one that exposes
   the EfficientNetV2 backbone's `top_activation` layer (before MSAM) and one
   that exposes the last `MSAM` layer (after MSAM). Gradients of the
   predicted class w.r.t. each are used to produce heatmap overlays, saved to
   `static/` and displayed side by side on the result page.

## Notes / things to double check

- **Class order**: `utils/predict.py` hard-codes the 23 class names in the
  exact alphabetical order produced by
  `tf.keras.preprocessing.image_dataset_from_directory` in the training
  notebook. If you retrain on a different folder structure, regenerate this
  list from `train_ds.class_names`.
- **Custom layers**: `ECABlock` and `MSAM` in `utils/gradcam.py` are copied
  verbatim from the training notebook (same default arguments — `filters=256`,
  `kernel_size=3`) so the saved `.keras` weights load correctly.
- This is a research/educational demo, **not** a diagnostic medical device.
  The disclaimer shown in the app should always accompany any prediction.
