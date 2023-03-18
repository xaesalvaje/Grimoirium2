from scripts.create_master_corpus import create_master_corpus
import os
import logging
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors

# Load the pre-trained GloVe model
glove_model_path = "./data/glove/glove.840B.300d.txt"
glove_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=False)

# Use the pre-trained model as the vocabulary for your own model
my_model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
my_model.build_vocab_from_freq(glove_model.vocab)
my_model.intersect_word2vec_format(glove_model_path, binary=False, lockf=1.0)
my_model.train(sentences, total_examples=my_model.corpus_count, epochs=my_model_epochs)

def train_word2vec_model(sentences, model_path, model_epochs):
    logging.info("Training Word2Vec model...")
    model = Word2Vec(min_count=1, workers=4, sg=1)
    model.build_vocab() # Build vocabulary
    model.train(data/proccessed/merged.txt, total_examples=model.corpus_count, epochs=model_epochs)
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")

# Set the paths to the input and output folders
input_dir = "./data/output"
model_dir = "./models"
logs_dir = "./logs"

# Create the models directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Create the logs directory if it doesn't exist
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(logs_dir, "train_model.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define a callback to log the training progress
class LogCallback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        logging.info(f"Epoch {self.epoch} - Loss: {loss}")
        self.epoch += 1

# Get a list of preprocessed text files in the input directory
preprocessed_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

# Set up the Word2Vec model parameters
model_window = 5
model_min_count = 5
model_workers = 4
model_epochs = 50

# Initialize the Word2Vec model
model = Word2Vec(
    window=model_window,
    min_count=model_min_count,
    workers=model_workers,
    callbacks=[LogCallback()]
)

# Build the vocabulary from the preprocessed text files
sentences = []
for preprocessed_file in preprocessed_files:
    preprocessed_path = os.path.join(input_dir, preprocessed_file)
    with open(preprocessed_path, "r") as f:
        text = f.read()
    words = text.split()
    sentences.append(words)
logging.info(f"Loaded {len(sentences)} sentences from {len(preprocessed_files)} preprocessed files.")
model.build_vocab(sentences)
logging.info(f"Built vocabulary of {len(model.wv.key_to_index)} unique words.")

# Train the Word2Vec model
model.train(sentences, total_examples=model.corpus_count, epochs=model_epochs)

# Create the master corpus
create_master_corpus(model, os.path.join(model_dir, "master_corpus.txt"))

# Destroy the Word2Vec model
del model

logging.info("Training complete.")
