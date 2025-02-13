# 🎓 Lecture Summarizer 📚🎤

## Overview
The **Lecture Summarizer** is an AI-powered tool that extracts **audio** and **visual** content from lecture videos, transcribes spoken words, performs **OCR on board content**, and combines the extracted text into a **structured summary**. The summarized content is then formatted into a **well-organized PDF** using an **LLM-based approach**.

<img width="1512" alt="image" src="https://github.com/user-attachments/assets/0129a9a7-139f-43b6-91f1-5dd667d5311d" />


## 🔥 Features
- 🎤 **Speech-to-Text Transcription**: Extracts and transcribes spoken content using Whisper ASR.
- 🖼️ **OCR for Board Content**: Extracts text from slides, whiteboards, and handwritten notes.
- 🤖 **LLM-Based Summarization**: Processes extracted text using an AI model (Gemini Pro) to generate structured notes.
- 📄 **PDF Generation**: Converts the summarized content into a **concise** and **well-formatted PDF** with key points.
- 🎯 **Efficient Text Processing**: Uses **text chunking** and **intelligent summarization** to limit the output to ~3 pages.

## 📂 Project Structure
```
Lecture-Summarizer/
│── Audio_Processing/
│   │── Convert.py  # Converts lecture videos to audio
│   │── Helper.py   # Handles transcription logic
│   │── Transcribe.py # Main transcription module
│
│── Video_Processing/
│   │── Main_keyframes.py  # Extracts keyframes from video
│   │── OCR_Helper.py      # Performs OCR on extracted keyframes
│
│── Summarization/
│   │── Summary.py  # Generates structured summaries using LLM
│
│── templates/
│   │── index.html
│
│── requirements.txt  # Project dependencies
│── README.md         # Project documentation
│── APP.py           # Main script to run the summarization pipeline
```
## 🚀 Installation & Setup
1. Clone the repository:
```sh
git clone https://github.com/Neerajjv/Lecture-Summarizer.git
cd Lecture-Summarizer	
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Run the Lecture Summarizer:
```
python APP.py
```

<img width="1512" alt="image" src="https://github.com/user-attachments/assets/915d5bcd-4823-4ffd-bdcc-6c87ebd3b133" />


## 🛠️ How It Works

### 1. 🎥 Extracts Audio & Visual Content:
- Converts lecture videos into audio.
- Extracts keyframes from the video.

### 2. 📝 Text Extraction:
- Uses Whisper ASR to transcribe spoken content.
- Runs OCR on keyframes to extract text from slides/boards.

### 3. 📚 Summarization:
- Merges both texts.
- Summarizes it using LLM (Gemini Pro).
- Ensures concise notes (~3 pages max).

### 4. 📄 PDF Generation:
- Formats the summary into a structured PDF.

## 🏆 Results:
- Organized lecture notes with headings, bullet points, and key takeaways.
- PDF output for easy sharing and reference.

## 📝 Future Improvements:
- 🎭 **Speaker Identification** for multi-speaker transcription.
- 📊 **Diagram & Chart Recognition** for enhanced visual content extraction.
- 🧠 **More LLMs Support** like GPT-4 and Claude for better summaries.

## 👨‍💻 Contributions & Feedback
Feel free to contribute, report issues, or suggest improvements! 🚀

📧 Contact
For queries, reach out or create an issue on GitHub.
