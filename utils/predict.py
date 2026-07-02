"""
predict.py
----------
Loads the trained EfficientNetV2-S + MSAM (balanced) model and runs
inference on a single uploaded image.
"""

import os
import numpy as np
import tensorflow as tf

from utils.gradcam import CUSTOM_OBJECTS

IMG_SIZE = (224, 224)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "best_balanced.keras"
)

# Must match the class order produced by
# tf.keras.preprocessing.image_dataset_from_directory(train_dir).class_names
# in the training notebook (alphabetical directory order).
CLASS_NAMES = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections",
]

_model = None


def get_model():
    """Lazily load and cache the Keras model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Place best_balanced.keras inside the 'model/' folder."
            )
        _model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
    return _model


def load_and_preprocess_image(image_path):
    """
    Returns:
      img_batch_0_255: (1, 224, 224, 3) float32, raw pixel range 0-255
                        (the model was trained without manual rescaling;
                        EfficientNetV2 applies its own internal preprocessing)
      img_0_1: (224, 224, 3) float32, pixel range 0-1 (used as the Grad-CAM
               background image)
    """
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)  # (224, 224, 3), 0-255

    img_batch_0_255 = np.expand_dims(img_array, axis=0).astype(np.float32)
    img_0_1 = (img_array / 255.0).astype(np.float32)

    return img_batch_0_255, img_0_1


def predict_top_k(img_batch_0_255, top_k=3):
    """Runs inference and returns the top-k (class_name, confidence%) pairs
    plus the full raw prediction vector."""
    model = get_model()
    preds = model.predict(img_batch_0_255, verbose=0)[0]

    top_indices = np.argsort(preds)[::-1][:top_k]
    results = [
        {"label": CLASS_NAMES[i], "confidence": float(preds[i] * 100)}
        for i in top_indices
    ]

    return results, preds
