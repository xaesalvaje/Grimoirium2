import os
import sys
import logging
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from scripts.train_model import train_word2vec_model
from scripts.combine_pdf import merge_pdfs
from scripts.convert_to_text import convert_to_txt
from scripts.create_master_corpus import create_master_corpus
from nltk.tokenize import sent_tokenize

def preprocess_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)

            with open(input_file_path, "r") as input_file:
                text = input_file.read()

                # Tokenize text into sentences
                sentences = sent_tokenize(text)

            with open(output_file_path, "w") as output_file:
                for sentence in sentences:
                    # Write each sentence on a new line
                    output_file.write(sentence + "\n")

# Set up file paths
input_dir = "data/raw"
output_dir = "data/processed"
processed_dir = "data/processed/"
raw_dir = "data/raw/"
logs_dir = "./logs"

# Create directories if they don't exist
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Set up NLTK resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Set up logging
logging.basicConfig(
    filename=os.path.join(logs_dir, "preprocess_text.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


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

# Merge PDF files into one
merged_pdf = merge_pdfs(input_dir, output_dir)
print("PDF files merged successfully!")

# Convert merged PDF into a text file
convert_to_txt(output_dir, processed_dir)

# Preprocess the text files
text_files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]
for text_file in text_files:
    input_path = os.path.join(processed_dir, text_file)
    output_path = os.path.join(raw_dir, text_file)
    try:
        with open(input_path, "r") as f:
            text = f.read()
        preprocessed_text = preprocess_text(text)
        with open(output_path, "w") as f:
            f.write(preprocessed_text)
        logging.info(f"Preprocessed {text_file}")
    except Exception as e:
        logging.error(f"Error processing {text_file}: {e}")
        
# Preprocess the text files
text_files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]
for text_file in text_files:
    with open(os.path.join(processed_dir, text_file), "r") as f:
        text = f.read()
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stop words
        tokens = [token for token in tokens if token.lower() not in stop_words]
        # Perform stemming
        tokens = [porter_stemmer.stem(token) for token in tokens]
        # Add processed text to corpus
        corpus.append(tokens)

# Train the LDA model
dictionary = corpora.Dictionary(corpus)
doc_term_matrix = [dictionary.doc2bow(tokens) for tokens in corpus]
lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary, passes=passes)

# Save the LDA model
lda_model.save(model_file)

# Print the topics
topics = lda_model.print_topics(num_words=num_words)
for topic in topics:
    print(topic)
    