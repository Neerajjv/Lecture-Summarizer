from pydub import AudioSegment
import os

def convert_video_to_audio(video_path: str, output_format="wav") -> str:
    """
    Converts a video file to an audio file using pydub.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found.")

    # Define the output path
    audio_output_path = f"{os.path.splitext(video_path)[0]}.{output_format}"

    # Convert video to audio
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_output_path, format=output_format)

    return audio_output_path