# predictor.py

import pickle
from keras.models import load_model
from utils import load_tokenizer, final_model

# --- Parameters ---
model_path = "Model/glove6B100D_lstm.h5"
tokenizer_path = "Model/tokenizer.pkl"
max_seq_len = 24  # Set to the value used during training

# --- Load model and tokenizer ---
print("[INFO] Loading model and tokenizer...")
model = load_model(model_path)
tokenizer = load_tokenizer(tokenizer_path)

# --- Interactive CLI ---
if __name__ == "__main__":
    print("Type a seed sentence (type 'exit' to quit):")
    while True:
        seed = input("You: ")
        if seed.lower() == "exit":
            break
        result = final_model(seed, model, tokenizer, max_seq_len, num_words=1)
        print("→ Prediction:", result)
