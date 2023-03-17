import os
import PyPDF2
import textract

def get_stopwords(stopwords_file):
    """Gets stopwords from file"""
    with open(stopwords_file, 'r') as f:
        stopwords = f.read().split()
    return stopwords

def extract_text_from_pdf(filename):
    """Extracts text from PDF file"""
    with open(filename, 'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        text = ''
        for i in range(pdf_reader.numPages):
            page = pdf_reader.getPage(i)
            text += page.extractText()
        return text

def extract_text_from_docx(filename):
    """Extracts text from DOCX file"""
    text = textract.process(filename)
    return text.decode('utf-8')

def get_files_from_directory(directory):
    """Gets list of files from directory"""
    files = []
    for file in os.listdir(directory):
        if file.endswith('.pdf') or file.endswith('.docx'):
            files.append(os.path.join(directory, file))
    return files
