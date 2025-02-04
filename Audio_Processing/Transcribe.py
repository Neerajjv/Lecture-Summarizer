import os
from Audio_Processing.Helper import transcribe
from Audio_Processing.Convert import convert_video_to_audio

def Transcribe(video_file, output_text_file):  # Accepts 2 parameters now

    # Step 1: Convert video to audio
    try:
        audio_file = convert_video_to_audio(video_file)
        print(f"Audio extracted successfully: {audio_file}")
    except Exception as e:
        print(f"Error converting video to audio: {e}")
        return None

    # Step 2: Transcribe the audio
    try:
        transcription = transcribe(audio_file, task="transcribe", output="txt")
        
        # Save transcription to a text file
        with open(output_text_file, "w", encoding="utf-8") as file:
            file.write(transcription)
        
        print(f"Transcription saved to {output_text_file}")
        return output_text_file
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None