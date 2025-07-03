# predictor.py

import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --- Parameters ---
model_path = "Model/glove6B100D_lstm.h5"
tokenizer_path = "Model/tokenizer.pkl"
max_seq_len = 23  # Set this to the max_seq_len-1 used during training

# --- Load model and tokenizer ---
print("[INFO] Loading model and tokenizer...")
model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# --- Prediction Function ---
def predict_next_word(seed_text, num_words=1):
    for _ in range(5):
        for _ in range(num_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted_index = np.argmax(predicted_probs)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            if output_word == "":
                break
            seed_text += " " + output_word
    return seed_text

# --- Interactive CLI ---
if __name__ == "__main__":
    print("Type a seed sentence (type 'exit' to quit):")
    while True:
        seed = input("You: ")
        if seed.lower() == "exit":
            break
        result = predict_next_word(seed, num_words=1)
        print("→ Prediction (next 5 words):", result)
