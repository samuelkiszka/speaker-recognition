import io
import numpy as np
import librosa

def extract_mfcc(file_bytes, sr=16000, n_mfcc=13):
    """
    Extracts MFCC features along with delta and delta-delta (derivatives).
    Returns 39-dimensional features per frame.
    """
    with io.BytesIO(file_bytes) as f:
        y, orig_sr = librosa.load(f, sr=sr, mono=True)

    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    # Base MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # First derivative (delta)
    delta = librosa.feature.delta(mfcc)

    # Second derivative (delta-delta)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack: shape (39, frames)
    combined = np.vstack([mfcc, delta, delta2])
    return combined.T  # (frames, 39)
