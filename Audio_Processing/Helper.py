import os
from io import StringIO
from threading import Lock
from typing import BinaryIO, Union
import torch
import whisper
from whisper.utils import WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT


import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

with urllib.request.urlopen("https://example.com") as response:
    content = response.read()

# Initialize Whisper Model
model_name = os.getenv("ASR_MODEL", "base")
model_path = os.getenv("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))


if torch.cuda.is_available():
    model = whisper.load_model(model_name, download_root=model_path).cuda()
else:
    model = whisper.load_model(model_name, download_root=model_path)
model_lock = Lock()


def transcribe(audio, task=None, language=None, initial_prompt=None, word_timestamps=None, output="txt"):
    """
    Transcribes the audio using the Whisper model.
    """
    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if word_timestamps:
        options_dict["word_timestamps"] = word_timestamps

    with model_lock:
        result = model.transcribe(audio, **options_dict)

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file.read()


def language_detection(audio):
    """
    Detects the language of the given audio.
    """
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    with model_lock:
        _, probs = model.detect_language(mel)

    return max(probs, key=probs.get)


from whisper.utils import WriteSRT, WriteVTT, WriteTSV, WriteJSON, WriteTXT, ResultWriter

def write_result(result: dict, file: BinaryIO, output: str):
    """
    Writes transcription result to the specified output format.
    """
    options = {"max_line_width": 1000, "max_line_count": 10, "highlight_words": False}
    
    # Create a ResultWriter instance (passing None for output_dir as we're writing to memory, not a directory)
    result_writer = ResultWriter(output_dir=None)

    if output == "srt":
        WriteSRT(result_writer).write_result(result, file=file, options=options)
    elif output == "vtt":
        WriteVTT(result_writer).write_result(result, file=file, options=options)
    elif output == "tsv":
        WriteTSV(result_writer).write_result(result, file=file, options=options)
    elif output == "json":
        WriteJSON(result_writer).write_result(result, file=file, options=options)
    elif output == "txt":
        WriteTXT(result_writer).write_result(result, file=file, options=options)
    else:
        raise ValueError(f"Unsupported output format: {output}")