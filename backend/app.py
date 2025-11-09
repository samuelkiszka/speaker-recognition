from flask import Flask, request, jsonify
from pydub import AudioSegment
import io
from flask_cors import CORS
from pathlib import Path
import numpy as np
import json

from utils import features, embedding_model, matching, asr

app = Flask(__name__)
CORS(app)

DATA_DIR = Path("data")
EMB_DIR = DATA_DIR / "embeddings"
SPEAKER_FILE = DATA_DIR / "speakers.json"

EMB_DIR.mkdir(parents=True, exist_ok=True)
if not SPEAKER_FILE.exists():
    SPEAKER_FILE.write_text("[]")


@app.route("/register", methods=["POST"])
def register_speaker():
    """Registrace mluvčího – přijme jméno a audio"""
    name = request.form.get("name")
    file = request.files.get("file")
    if not name or not file:
        return jsonify({"error": "Missing name or file"}), 400

    # Convert any format to WAV using pydub
    audio_bytes = file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    # Extract MFCC from WAV bytes
    mfcc = features.extract_mfcc(wav_io.read())
    emb = embedding_model.get_embedding(mfcc)

    # Save embedding
    emb_path = EMB_DIR / f"{name}.npy"
    np.save(emb_path, emb)

    # Update speaker database
    speakers = json.loads(SPEAKER_FILE.read_text())
    speakers.append({"name": name, "path": str(emb_path)})
    SPEAKER_FILE.write_text(json.dumps(speakers, indent=2))

    return jsonify({"status": "ok", "speaker": name})


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Transkripce a rozpoznání mluvčích"""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "Missing file"}), 400

    # Convert any format to WAV using pydub
    audio_bytes = file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)


    segments = asr.transcribe_and_segment(wav_io.read())

    speakers = json.loads(SPEAKER_FILE.read_text())
    known_embs = {s["name"]: np.load(s["path"]) for s in speakers}

    for seg in segments:
        mfcc = features.extract_mfcc(seg["audio"])
        emb = embedding_model.get_embedding(mfcc)
        seg["speaker"] = matching.find_best_match(emb, known_embs)
        seg.pop("audio", None)  # remove raw bytes

    print(segments)

    return jsonify({"segments": segments})


@app.route("/speakers", methods=["GET"])
def list_speakers():
    speakers = json.loads(SPEAKER_FILE.read_text())
    return jsonify(speakers)


@app.route("/speakers/<name>", methods=["DELETE"])
def delete_speaker(name):
    speakers = json.loads(SPEAKER_FILE.read_text())
    speakers = [s for s in speakers if s["name"] != name]
    SPEAKER_FILE.write_text(json.dumps(speakers, indent=2))
    emb_path = EMB_DIR / f"{name}.npy"
    if emb_path.exists():
        emb_path.unlink()
    return jsonify({"status": "deleted", "speaker": name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
