from Video_Processing.Main_keyframes import process_video_and_cluster
from Audio_Processing.Transcribe import Transcribe
from Video_Processing.OCR_Helper import extract_text_from_images
from Summarization.Summary import generate_summarized_pdf
import os

# Define parameters
video_path = "test.mp4"  # Video path
frame_rate = 5  # 1 frame every 5 seconds
output_slides_dir = "/Transcript/output"
output_text_file = "/Transcript/Result/results.txt"  # Text extracted from video
audio_transcription_file = "/Transcript/Result/audio_transcription.txt"  # Transcribed audio text
final_output_file = "/Transcript/Result/Final_Combined.txt"  # Merged text file
summary_pdf_file = "/Transcript/Result/Summary_Report.pdf"  # Summarized PDF file

# Ensure output directories exist
os.makedirs(os.path.dirname(output_text_file), exist_ok=True)

# Step 1: Process video frames and extract text
output_dirs = process_video_and_cluster(video_path, frame_rate, output_slides_dir, output_text_file)
extract_text_from_images(output_dirs["representative_frames"], output_text_file)
print(f"Extracted text from frames saved in: {output_text_file}")

# Step 2: Process audio transcription
Transcribe(video_path, audio_transcription_file)
print(f"Transcribed audio text saved in: {audio_transcription_file}")

# Step 3: Combine both extracted texts into a single file
try:
    with open(output_text_file, "r", encoding="utf-8") as video_text_file:
        video_text = video_text_file.read().strip()

    with open(audio_transcription_file, "r", encoding="utf-8") as audio_text_file:
        audio_text = audio_text_file.read().strip()

    # Format and save the combined content
    with open(final_output_file, "w", encoding="utf-8") as final_file:
        final_file.write(f"Video_Extraction:\n{video_text}\n\n\n\n\n")  # 5 lines below
        final_file.write(f"Audio_Extraction:\n{audio_text}")

    print(f"Final combined text saved in: {final_output_file}")

except Exception as e:
    print(f"Error combining text files: {e}")

# Step 4: Summarize the final combined text and save it as a PDF
generate_summarized_pdf(final_output_file, summary_pdf_file)
print(f"Summarized report saved as PDF: {summary_pdf_file}")