"""
DeepDetect Sentinel — Flask Backend
Serves the forensic analysis UI and handles video upload + inference.
"""

import os
import json
import time
import uuid
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload
app.config["UPLOAD_FOLDER"] = os.path.join(tempfile.gettempdir(), "deepdetect_uploads")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Import model (lazy loading)
from model import analyze_video

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ─────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main SPA."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze a single uploaded video."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        ext = os.path.splitext(file.filename)[1]
        return jsonify({
            "error": f"Unsupported format: {ext}. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save to temp location
    safe_name = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    try:
        file.save(filepath)
        start_time = time.time()
        result = analyze_video(filepath)
        elapsed = round(time.time() - start_time, 2)

        result["filename"] = file.filename
        result["processing_time"] = elapsed

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """Analyze multiple videos in batch."""
    files = request.files.getlist("videos")
    if not files:
        return jsonify({"error": "No video files provided"}), 400

    results = []
    for file in files:
        if file.filename == "" or not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": "Unsupported or invalid file",
                "prediction": None,
                "prob_fake": None,
            })
            continue

        safe_name = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

        try:
            file.save(filepath)
            start_time = time.time()
            result = analyze_video(filepath)
            elapsed = round(time.time() - start_time, 2)

            result["filename"] = file.filename
            result["processing_time"] = elapsed
            results.append(result)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "prediction": None,
                "prob_fake": None,
            })
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({"results": results})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "nominal", "version": "Sentinel-Core-88.39"})


# ─── Entry Point ────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  DeepDetect Sentinel — Forensic Lens v3.4   ║")
    print("║  Starting analysis server...                ║")
    print("╚══════════════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=5000, debug=True)
