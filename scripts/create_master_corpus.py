from gensim.models import Word2Vec
from scripts.preprocess_text import preprocess_text

def create_master_corpus(input_file, output_file):
    # preprocess the text in the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    preprocessed_text = preprocess_text(text)

    # train the Word2Vec model
    model = Word2Vec(sentences=preprocessed_text, size=100, window=5, min_count=1, workers=4)

    # write the learned vocabulary to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in model.wv.index_to_key:
            f.write(word + '\n')

    # destroy the model to free up memory
    del model
