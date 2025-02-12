import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq  # Groq integration
import warnings
from fpdf import FPDF  # Import the FPDF library for PDF generation

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".LangChainDeprecationWarning.")


def generate_summarized_pdf(txt_path, output_pdf_path):
    """
    Generates a summarized PDF from a given text file, ensuring a maximum of 3 pages.

    Parameters:
    - txt_path (str): Path to the input text file.
    - output_pdf_path (str): Path to save the output PDF.
    """
    GROQ_API_KEY = "YOUR_API_KEY"  # Groq API Key

    # Function to load data from a text file
    def load_data_from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    # Function to split text into chunks (Larger chunks for concise summaries)
    def text_split(data):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Increased chunk size
        return splitter.split_text(data)

    # Function to generate a structured, concise summary using Groq
    def generate_summary(chunk):
        prompt = f"""
        Summarize the following text **very concisely**, ensuring the final summary does not exceed 3 pages when compiled:
        Text: {chunk}

        **Format the summary as follows:**
        - **Summary:** Provide a **2-3 sentence** summary of the text.
        - **Key Points:** List **only the top 2-3 key points** in short bullet points.

        Keep the summary brief, removing unnecessary details while retaining the **main idea**.
        """

        # Initialize Groq API client
        llm = ChatGroq(
            temperature=0.3,
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile"  # Model for summarization
        )

        # Call Groq API to generate summary
        try:
            response = llm.invoke(prompt)
            if response and hasattr(response, "content"):
                response_text = response.content
                summary_text, key_points_text = response_text.split(
                    "**Key Points**:") if "**Key Points**:" in response_text else (response_text, "")
                return f"**Summary:**\n{summary_text.strip()}\n\n**Key Points:**\n{key_points_text.strip()}"
        except Exception as e:
            return f"Error in generating summary: {str(e)}"

        return "Error in generating summary."

    # Load the text data from the file
    docs = load_data_from_file(txt_path)

    # Split text into chunks (for better summarization)
    text_chunks = text_split(docs)

    # Generate summaries for all chunks using Groq
    summarized_notes = [f"### Section {i + 1}\n{generate_summary(chunk)}\n" for i, chunk in enumerate(text_chunks)]

    # Create a PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set font for the PDF
    pdf.set_font("Arial", size=12)

    # Title for the PDF
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, txt="Summarized Notes with Key Points", ln=True, align="C")
    pdf.ln(10)

    # Add structured summary content to the PDF
    for i, section in enumerate(summarized_notes):
        if pdf.page_no() >= 3:  # Stop adding content after page 3
            break
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(0, 10, f"Section {i + 1}", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, section)

    # Save the PDF
    pdf.output(output_pdf_path)
    print(f"Summarized notes have been saved to {output_pdf_path}")

