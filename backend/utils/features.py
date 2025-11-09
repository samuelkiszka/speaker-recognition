import io
import numpy as np
import librosa
import soundfile as sf

def extract_mfcc(audio_bytes, sr=16000, n_mfcc=13):
    """Extrahuje MFCC příznaky z WAV souboru (v bytech)."""
    y, orig_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T
