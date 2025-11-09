import wave
import json
import os
from vosk import Model, KaldiRecognizer

# cesta k modelu (uprav podle staženého modelu)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/vosk-model-small-cs-0.4-rhasspy")
model = Model(MODEL_PATH)


def transcribe_and_segment(file_bytes):
    """
    Přepis WAV souboru a návrat segmentů s časovou informací.

    Args:
        file_bytes: bytes z Flask request.files

    Returns:
        List[dict]: [{'start': float, 'end': float, 'text': str, 'audio': bytes}]
    """
    # Uložme bytes do dočasného wav souboru
    import io
    temp_wav = io.BytesIO(file_bytes)

    # Otevřeme s wave modul
    wf = wave.open(temp_wav, "rb")
    if wf.getnchannels() != 1:
        raise ValueError("Vosk vyžaduje mono audio.")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    segments = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            print(res)
            for word in res.get("result", []):
                print(word["word"])
                segments.append({
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"],
                    "audio": None  # audio není nutné ukládat, ale může se nahradit originálem
                })

    # doplníme i poslední část
    res = json.loads(rec.FinalResult())
    for word in res.get("result", []):
        segments.append({
            "start": word["start"],
            "end": word["end"],
            "text": word["word"],
            "audio": None
        })

    wf.close()
    return segments
