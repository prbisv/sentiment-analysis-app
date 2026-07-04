# 🚚 Indonesian Delivery Service Sentiment Analysis using CNN-LSTM

A web-based sentiment analysis application for Indonesian delivery service reviews using a Convolutional Neural Network - Long Short-Term Memory (CNN-LSTM) model.

---

## 📌 Overview

This project classifies Indonesian delivery service reviews into three sentiment categories:

- 😊 Positive
- 😐 Neutral
- ☹️ Negative

The application provides:

- Sentiment prediction
- Prediction confidence
- Probability for each class
- Preprocessed text visualization

The model is served through a Flask web application with a clean and responsive user interface.

---

## ✨ Features

- CNN-LSTM Deep Learning model
- Indonesian text preprocessing
- Slang word normalization
- KBBI dictionary filtering
- Stopword removal
- Indonesian stemming (Sastrawi)
- Confidence score
- Probability visualization
- Input validation
- Responsive UI

---

## 🧠 Model Architecture

Embedding

↓

Conv1D (100 Filters, Kernel Size = 5)

↓

Conv1D (64 Filters, Kernel Size = 3)

↓

LSTM (128 Units)

↓

Global Max Pooling

↓

Dropout (0.5)

↓

Dense (Softmax)

---

## 📂 Project Structure

```text
sentiment-analysis-app/

├── app.py
├── requirements.txt
├── model/
│   ├── cnn_lstm_model3_84_earlystopping_new.keras
│   ├── tokenizer.pkl
│   ├── config.json
│   └── labels.json
│
├── resources/
│   ├── dataKBBI.json
│   └── slang_word_update.csv
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
└── utils/
    ├── preprocessing.py
    └── predict.py
```

---

## ⚙️ Technologies

- Python
- Flask
- TensorFlow / Keras
- NumPy
- NLTK
- Sastrawi
- HTML
- CSS

---

## 🚀 Installation

Clone this repository

```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
```

Move into the project directory

```bash
cd sentiment-analysis-app
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 🖥️ Application Preview

<img width="859" height="572" alt="Screenshot 2026-07-05 at 06 53 05" src="https://github.com/user-attachments/assets/a6d14e75-0c2a-4672-935b-09e391fff94d" />

---

## 📈 Prediction Example

Input

```
Kurir sangat ramah dan paket datang lebih cepat.
```

Prediction

```
😊 Positive
Confidence : 97.82%
```

---

## 📄 Dataset

The dataset consists of Indonesian delivery service reviews that have been manually labeled into:

- Positive
- Neutral
- Negative

---

## 👩‍💻 Author

**Prabhaisvari Sadhaka**

GitHub:
https://github.com/prbisv

---

## 📜 License

This project is created for educational and research purposes.
