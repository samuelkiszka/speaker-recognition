def transcribe_and_segment(audio_bytes):
    """
    Zde by normálně běžel Vosk nebo Whisper.
    Zatím placeholder vrací dva segmenty s fiktivními texty.
    """
    return [
        {"start": 0.0, "end": 2.0, "text": "Dobrý den.", "audio": audio_bytes},
        {"start": 2.0, "end": 4.0, "text": "Jak se máte?", "audio": audio_bytes},
    ]
