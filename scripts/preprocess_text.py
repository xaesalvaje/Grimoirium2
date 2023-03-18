import os
import logging
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set the paths to the input and output folders
input_dir = "./data/input"
output_dir = "./data/output"
logs_dir = "./logs"

# Create the logs directory if it doesn't exist
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(logs_dir, "preprocess_text.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set up NLTK resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Define the preprocessing function
def preprocess_text(text):
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
    return preprocessed_text

# Get a list of text files in the input directory
text_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

# Loop through each text file and preprocess the text
for text_file in text_files:
    input_path = os.path.join(input_dir, text_file)
    output_path = os.path.join(output_dir, text_file)
    try:
        with open(input_path, "r") as f:
            text = f.read()
        preprocessed_text = preprocess_text(text)
        with open(output_path, "w") as f:
            f.write(preprocessed_text)
        logging.info(f"{text_file} preprocessed successfully.")
    except Exception as e:
        logging.error(f"Error preprocessing {text_file}: {e}")
