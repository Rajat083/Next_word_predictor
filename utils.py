import re
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Text Preprocessing
# ----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?.!,']+", " ", text)
    text = re.sub(r"[()]", "", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\,{2,}", ",", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Tokenizer Functions
# ----------------------------

def save_tokenizer(tokenizer, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# ----------------------------
# Sequence Generation
# ----------------------------

def create_ngrams(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    ngram_sequences = []

    for seq in sequences:
        for i in range(1, len(seq)):
            ngram_seq = seq[:i + 1]
            ngram_sequences.append(ngram_seq)

    max_seq_len = max(len(seq) for seq in ngram_sequences)
    input_sequences = pad_sequences(ngram_sequences, maxlen=max_seq_len, padding='pre')
    return input_sequences, max_seq_len

# ----------------------------
# Final Model Interface
# ----------------------------

def final_model(seed_text, model, tokenizer, max_seq_len, num_words=1):
    """
    Predicts next word(s) using trained model.
    - seed_text: input sentence
    - model: trained Keras model
    - tokenizer: fitted tokenizer
    - max_seq_len: padding length used during training
    - num_words: number of words to predict iteratively
    """
    for _ in range(num_words):
        for _ in range(5):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
            predicted = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=-1)[0]

            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    output_word = word
                    break

            if output_word:
                seed_text += " " + output_word
            else:
                break

    return seed_text
