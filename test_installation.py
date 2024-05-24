# test_installation.py
try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    from speechbrain.pretrained import SpeakerRecognition
    print("All packages imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
