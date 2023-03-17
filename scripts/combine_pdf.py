import os
import re
import logging
from io import StringIO
from PyPDF2 import PdfFileReader, PdfFileMerger
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def merge_pdfs(input_dir, output_path):
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    pdf_files.sort()
    pdf_merger = PdfFileMerger()

    for pdf_file in pdf_files:
        pdf_merger.append(PdfFileReader(os.path.join(input_dir, pdf_file), "rb"))

    with open(output_path, "wb") as outfile:
        pdf_merger.write(outfile)

    print(f"Successfully merged PDFs into {output_path}")

    # Convert PDFs to text files
    convert_to_txt(output_path, input_dir)

def convert_to_txt(input_file, output_dir):
    with open(input_file, 'rb') as fp:
        rsrcmgr = PDFResourceManager()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, StringIO(), codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
            text = device.get_text()
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0] + '.txt')
            with open(output_path, "w") as f:
                f.write(text)
            logging.info(f"{input_file} converted to text successfully.")

    # Preprocess text files
    preprocess_text(input_dir, output_dir)

def preprocess_text(input_dir, output_dir):
    # Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            # Read the text file
            with open(os.path.join(input_dir, filename), 'r') as f:
                text = f.read()

            # Convert to lowercase
            text = text.lower()

            # Remove non-alphanumeric characters and multiple whitespaces
            text = re.sub(r"[^a-z0-9\s]+", "", text)
            text = re.sub(r"\s+", " ", text)

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stop words
            tokens = [t for t in tokens if t not in stop_words]

            # Lemmatize the tokens
            tokens = [lemmatizer.lemmatize(t) for t in tokens]

            # Join the tokens back into a string
            preprocessed_text = " ".join(tokens)

            # Write the preprocessed text to the output directory
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(preprocessed_text)

            logging.info(f"{filename} preprocessed successfully.")

# Set the paths to the input and output folders
input_dir = "./data/input"
output_dir = "./data/output"

# Merge the PDF files into a single file
output_path = os.path.join(output_dir, "merged.pdf")
merge_pdfs(input_dir, output_path)

print("PDF files merged successfully!")

# Convert PDF files to text
def convert_to_txt(input_dir, output_dir):
    for pdf_file in os.listdir(input_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, pdf_file)
            output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
            
            with open(pdf_path, 'rb') as fp, open(output_path, "w") as f:
                text = ""
                pdf_reader = PdfReader(fp)
                
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # Write the extracted text to the output file
                f.write(text)
            
            print(f"{pdf_file} converted to text successfully.")
            
# Set the paths to the input and output folders
input_dir = "./data/input"
output_dir = "./data/output"

# Convert PDF files to text
convert_to_txt(input_dir, output_dir)

print("PDF files converted to text successfully!")

# Preprocess text files
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(input_file_path, output_file_path):
    # Read the input file
    with open(input_file_path, "r") as f:
        text = f.read()
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphanumeric characters and multiple whitespaces
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    # Join the tokens back into a string
    preprocessed_text = " ".join(tokens)
    
    # Write the preprocessed text to the output file
    with open(output_file_path, "w") as f:
        f.write(preprocessed_text)
    
    print(f"{input_file_path} preprocessed successfully.")

# Set the paths to the input and output folders
input_dir = "./data/output"
output_dir = "./data/processed"

# Preprocess text files
for file_name in os.listdir(input_dir):
    input_file_path = os.path.join(input_dir, file_name)
    output_file_path = os.path.join(output_dir, file_name)
    
    # Preprocess the text file
    preprocess_text(input_file_path, output_file_path)

print("Text files preprocessed successfully!")