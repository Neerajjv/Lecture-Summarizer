import os
from Helper import transcribe
from Convert import convert_video_to_audio


def Transcribe(video_file): #Pass Video Directory

    # Step 1: Convert video to audio
    try:
        audio_file = convert_video_to_audio(video_file)
        print(f"Audio extracted successfully: {audio_file}")
    except Exception as e:
        print(f"Error converting video to audio: {e}")
        return

    # Step 2: Transcribe the audio
    try:
        transcription = transcribe(audio_file, task="transcribe", output="txt")
        print("Transcription Output:")
        print(transcription)
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return

if __name__ == "__main__":
    Transcribe()