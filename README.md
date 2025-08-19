# 🔮 Next Word Predictor

A simple yet effective deep learning-based **Next Word Prediction** model built using LSTM in Keras. It learns from a custom text corpus and predicts the most likely next word(s) given a sentence prefix.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NLP](https://img.shields.io/badge/Topic-NLP-orange.svg)

---

## 🚀 Features

- Predicts the **most likely next word** from a given input sentence
- Built using **Natural Language Processing (NLP)** techniques
- Trained on a **corpus of English sentences**
- Tokenization and sequence modeling using **Keras / TensorFlow**
- Easily extendable to large datasets or character-level models

---

## 🧠 Technologies Used

- Python 3.10+
- TensorFlow / Keras 2.14.0
- NumPy
- Jupyter Notebook (for experiments)

---


## 🏁 Getting Started

🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Rajat083/Next_word_predictor.git
cd next-word-predictor
```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\\Scripts\\activate on Windowsll -r requirements.txt
    ```
3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

# Usage
Run the script
```bash
python predictor.py
```
Enter a sentence
```
You: {your_sentence}
→ Prediction: {your_sentence + next 5 predicted words}
```

## 🛠️ Model Details
Tokenizer: Top 5000 words, OOV token enabled

Model:

Embedding Layer

LSTM (128 units)

Dense output with softmax

Input Format: Padded n-gram sequences (using utils.create_ngrams)

Trained with: Categorical crossentropy, Adam optimizer

##Training is done inside the notebook:
```
next_word_predictor.ipynb
```
## 📚 Requirements
TensorFlow == 2.14

NumPy < 2

Keras (if standalone)
