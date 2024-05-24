# test_speechbrain_import.py
try:
    from speechbrain.pretrained import SpeakerRecognition
    print("SpeechBrain imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
