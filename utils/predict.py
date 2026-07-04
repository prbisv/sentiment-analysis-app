import json
import pickle
from pathlib import Path

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from utils.preprocessing import preprocess


# ==================================================
# PATH CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

# ==================================================
# LOAD MODEL
# ==================================================
model = load_model(
    MODEL_DIR / "cnn_lstm_model3_84_earlystopping_new.keras"
)

# ==================================================
# LOAD TOKENIZER
# ==================================================
with open(
    MODEL_DIR / "tokenizer.pkl",
    "rb"
) as f:

    tokenizer = pickle.load(f)

# ==================================================
# LOAD CONFIG
# ==================================================
with open(
    MODEL_DIR / "config.json",
    encoding="utf-8"
) as f:

    config = json.load(f)

MAXLEN = config["maxlen"]

# ==================================================
# LOAD LABELS
# ==================================================
with open(
    MODEL_DIR / "labels.json",
    encoding="utf-8"
) as f:

    LABELS = json.load(f)


# ==================================================
# MAIN FUNCTION
# ==================================================
def predict_sentiment(text):

    cleaned_text = preprocess(text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAXLEN,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0]

    predicted_class = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_class]
    )

    return {

        "input": text,

        "processed_text": cleaned_text,

        "label": LABELS[str(predicted_class)],

        "confidence": round(
            confidence * 100,
            2
        ),

        "probabilities": {

            LABELS["0"]: round(float(prediction[0]) * 100, 2),

            LABELS["1"]: round(float(prediction[1]) * 100, 2),

            LABELS["2"]: round(float(prediction[2]) * 100, 2)

        }

    }


# # ==========================================
# # TEST
# # ==========================================

# if __name__ == "__main__":

#     result = predict_sentiment(
#         "Kurirnya cepat sekali dan ramah"
#     )

#     print(result)