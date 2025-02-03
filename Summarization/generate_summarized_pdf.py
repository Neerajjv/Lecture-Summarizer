import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from fpdf import FPDF  # Import the FPDF library for PDF generation

# Suppress all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress LangChain-specific warnings
warnings.filterwarnings("ignore", message=".LangChainDeprecationWarning.")

# Retrieve API keys and configurations from environment variables
GEMINI_API_KEY = "Enter Your API key here"

def generate_summarized_pdf(txt_path, output_pdf_path="/content/summarized_notes_with_titles.pdf"):
    """
    Function to load text from a file, split it into chunks, generate summaries,
    and save the summaries as a PDF file with key points highlighted.

    Parameters:
    - txt_path (str): Path to the input text file.
    - output_pdf_path (str): Path to save the output PDF. Default is "/content/summarized_notes_with_titles.pdf".
    """

    # Function to load data from a text file
    def load_data_from_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read()
        return data

    # Function to split text into chunks
    def text_split(data):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = splitter.split_text(data)
        return text_chunks

    # Function to generate a structured summary for each chunk using the Gemini model
    def generate_summary(chunk):
        prompt = f"""
        Summarize the following text and highlight the key points in a structured format:
        Text: {chunk}
        Provide the summary with key points highlighted in bullet points. Format the summary as follows:
        1. *Summary*: [Summary of the text]
        2. *Key Points*:
           - Point 1: [Key Point 1]
           - Point 2: [Key Point 2]
           - Point 3: [Key Point 3]
        """

        # Request summary from the Gemini model
        response = llm.invoke(prompt)

        # Extract the actual text of the summary from the response
        summary_text = response.content.split("*Key Points*:")[0].strip()
        key_points_text = response.content.split("*Key Points*:")[-1].strip()

        # Clean and return the structured summary
        return f"*Summary:\n{summary_text}\n\nKey Points:*\n{key_points_text}"

    # Load the text data from the file
    docs = load_data_from_file(txt_path)

    # Split text into chunks
    text_chunks = text_split(docs)

    # Initialize Gemini API
    llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=GEMINI_API_KEY, temperature=0.7, top_p=0.85, max_tokens=500)

    # Generate summaries for all chunks
    summarized_notes = []
    for i, chunk in enumerate(text_chunks):
        summary = generate_summary(chunk)
        summarized_notes.append(f"### Section {i + 1}\n{summary}\n")

    # Output the structured summary with key points
    structured_summary = "\n".join(summarized_notes)

    # Create a PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set font for the PDF
    pdf.set_font("Arial", size=12)

    # Title for the PDF
    pdf.set_font("Arial", style='B', size=16)  # Bold and bigger font for title
    pdf.cell(200, 10, txt="Summarized Notes with Key Points", ln=True, align="C")

    # Add some space
    pdf.ln(10)

    # Set font for the body
    pdf.set_font("Arial", size=12)

    # Add the structured summary content to the PDF with bold headings for sections
    for i, section in enumerate(summarized_notes):
        # For each section, add a bold title (Section 1, Section 2, etc.)
        pdf.set_font("Arial", style='B', size=14)  # Bold font for section titles
        pdf.cell(0, 10, f"Section {i + 1}", ln=True)

        # Add the summary content with normal font
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, section)

    # Save the PDF to a file
    pdf.output(output_pdf_path)

    # Provide feedback to the user
    print(f"Summarized notes with titles have been saved to {output_pdf_path}")
