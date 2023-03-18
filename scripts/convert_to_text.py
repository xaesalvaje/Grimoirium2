import os
import logging
from pdfminer.layout import LAParams
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO

# Set the paths to the input and output folders
input_dir = "./data/input"
output_dir = "./data/output"

def convert_to_txt(input_dir, output_dir):
    # Get a list of PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    # Set up PDFResourceManager and TextConverter
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, StringIO(), codec=codec, laparams=laparams)

    # Loop through each PDF file and extract the text
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_path = os.path.join(output_dir, pdf_file.replace(".pdf", ".txt"))
        with open(pdf_path, 'rb') as fp:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.get_pages(fp):
                interpreter.process_page(page)
            text = device.get_text()

        with open(output_path, "w") as f:
            f.write(text)
        logging.info(f"{pdf_file} converted to text successfully.")
