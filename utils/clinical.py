"""
clinical.py
-----------
Loads clinical_module/clinical_info.json and exposes a lookup helper used
to attach descriptive medical context to the model's top prediction.
"""

import json
import os

_CLINICAL_INFO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "clinical_module",
    "clinical_info.json"
)

_clinical_info = None


def _load_clinical_info():
    global _clinical_info
    if _clinical_info is None:
        with open(_CLINICAL_INFO_PATH, "r", encoding="utf-8") as f:
            _clinical_info = json.load(f)
    return _clinical_info


def get_clinical_info(disease_name):
    """Returns the clinical info dict for a disease label, or None if the
    label isn't present in clinical_info.json."""
    info = _load_clinical_info()
    return info.get(disease_name)


DISCLAIMER = (
    "This prediction is intended for research and educational purposes only. "
    "It should not replace professional medical diagnosis."
)
