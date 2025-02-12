from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from Video_Processing.Main_keyframes import process_video_and_cluster
from Audio_Processing.Transcribe import Transcribe
from Video_Processing.OCR_Helper import extract_text_from_images
from Summarization.Summary import generate_summarized_pdf

app = Flask(__name__)

# Define upload and output directories
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")  # Upload form

@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No selected file"})

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Invalid file type"})

    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)

    # Define output paths
    output_slides_dir = os.path.join(RESULTS_FOLDER, "slides")
    output_text_file = os.path.join(RESULTS_FOLDER, "video_text.txt")
    audio_transcription_file = os.path.join(RESULTS_FOLDER, "audio_text.txt")
    final_output_file = os.path.join(RESULTS_FOLDER, "combined_text.txt")
    summary_pdf_file = os.path.join(RESULTS_FOLDER, "summary_report.pdf")

    # Step 1: Extract text from video frames
    output_dirs = process_video_and_cluster(video_path, frame_rate=5, output_slides_dir=output_slides_dir, output_text_file=output_text_file)

    # Step 2: Process audio transcription
    Transcribe(video_path, audio_transcription_file)

    # Step 3: Combine extracted text and transcribed text
    try:
        with open(output_text_file, "r", encoding="utf-8") as video_text_file, \
             open(audio_transcription_file, "r", encoding="utf-8") as audio_text_file:

            video_text = video_text_file.read().strip()
            audio_text = audio_text_file.read().strip()

        # Format and save the combined content
        with open(final_output_file, "w", encoding="utf-8") as final_file:
            final_file.write(f"Video Extraction:\n{video_text}\n\n\n\n\n")  # 5 lines below
            final_file.write(f"Audio Extraction:\n{audio_text}")

    except Exception as e:
        return jsonify({"success": False, "message": f"Error combining text files: {e}"})

    # Step 4: Summarize and generate a PDF
    generate_summarized_pdf(final_output_file, summary_pdf_file)

    return jsonify({"success": True, "message": "Processing complete!", "video_url": f"/uploads/{filename}"})

@app.route("/download")
def download_summary():
    summary_pdf_file = os.path.join(RESULTS_FOLDER, "summary_report.pdf")
    if os.path.exists(summary_pdf_file):
        return send_file(summary_pdf_file, as_attachment=True)
    return "Summary PDF not found."

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route("/view_summary")
def view_summary():
    summary_pdf_file = os.path.join(RESULTS_FOLDER, "summary_report.pdf")
    if os.path.exists(summary_pdf_file):
        return send_file(summary_pdf_file)  # Serve PDF instead of forcing download
    return "Summary PDF not found.", 404

@app.route("/check_summary")
def check_summary():
    summary_pdf_file = os.path.join(RESULTS_FOLDER, "summary_report.pdf")
    if os.path.exists(summary_pdf_file):
        return jsonify({"success": True})
    return jsonify({"success": False})


if __name__ == "__main__":
    app.run(debug=True)