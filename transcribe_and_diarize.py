import os
import subprocess
import sys
import torch
from whisper import load_model
from pyannote.audio import Pipeline
from huggingface_hub import login

# Function to extract audio from video
def extract_audio(video_file, audio_file, start_time, end_time):
    command = ['ffmpeg', '-i', video_file, '-q:a', '0', '-map', 'a']

    if start_time is not None:
        command.extend(['-ss', start_time])
    if end_time is not None:
        command.extend(['-to', end_time])

    command.append(audio_file)

    subprocess.run(command, check=True)

# Function to transcribe audio using Whisper model
def transcribe_audio(audio_file):
    model = load_model("large")
    result = model.transcribe(audio_file)
    return result['text'], result['segments']

# Function to perform speaker diarization using pyannote.audio
def perform_speaker_diarization(audio_file, token, num_speakers=4):
    login(token=token)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    diarization = pipeline(audio_file)

    # Get the speaker turns
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))

    return speaker_turns

# Main function
def main():
    video_file = '조별회의_PRGMS 3차 프로젝트_1팀 1조_4차.mp4'
    audio_file = 'audio.wav'

    start_time = None
    end_time = None
    num_speakers = 4

    if len(sys.argv) > 1:
        start_time = sys.argv[1] if sys.argv[1].lower() != 'none' else None
    if len(sys.argv) > 2:
        end_time = sys.argv[2] if sys.argv[2].lower() != 'none' else None
    if len(sys.argv) > 3:
        num_speakers = int(sys.argv[3])

    print("Starting audio extraction...")
    extract_audio(video_file, audio_file, start_time, end_time)
    print("Audio extraction completed")

    print("Loading Whisper model...")
    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    transcription, segments = transcribe_audio(audio_file)
    print("Transcription completed")

    print("Loading pyannote pipeline...")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    diarization = perform_speaker_diarization(audio_file, hf_token, num_speakers)
    print("Speaker diarization completed")

    print("Saving transcription with speaker labels...")
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        for start, end, speaker in diarization:
            segment_text = " ".join([word['text'] for word in segments if start <= word['start'] < end])
            f.write(f"Speaker {speaker}: {segment_text}\n")
    print("Transcription saved to transcription.txt")

if __name__ == "__main__":
    main()
