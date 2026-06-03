from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from flask_cors import CORS

from combined_preprocessing_pipeline import build_pipeline

app = Flask(__name__)
CORS(app)


@app.route("/process", methods=["POST"])
def process_text():
    payload = request.get_json(force=True)
    text = payload.get("text", "")
    modules = payload.get("modules", [])
    # Build kwargs for pipeline toggles
    kwargs: Dict[str, Any] = {k: True for k in modules}
    try:
        pipeline = build_pipeline(**kwargs)
        result = pipeline.process(text)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/config", methods=["GET"])
def config():
    # Expose available module keys for the frontend
    return jsonify({"modules": list(pipeline_available_keys())})


def pipeline_available_keys() -> List[str]:
    # Mirror the defaults from combined_preprocessing_pipeline
    try:
        from combined_preprocessing_pipeline import _MODULE_REGISTRY
        return list(_MODULE_REGISTRY.keys())
    except Exception:
        # Fallback if import fails
        return [
            "phonetic_normalization",
            "language_aware_normalization",
            "script_unification",
            "transliteration",
            "balanced_tokenization",
            "language_identification_tagging",
            "switch_point_encoding",
            "code_switch_augmentation",
            "context_aware_subword_sampling",
        ]


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
