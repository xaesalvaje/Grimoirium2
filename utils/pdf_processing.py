import io
import os
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """
    A function to extract text from a PDF file
    """
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ''
        for i in range(reader.getNumPages()):
            page = reader.getPage(i)
            text += page.extractText()

    return text
