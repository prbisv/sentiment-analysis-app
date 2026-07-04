import re

from flask import Flask, render_template, request

from utils.predict import predict_sentiment


# ==================================================
# INITIALIZATION
# ==================================================
app = Flask(__name__)


# ==================================================
# CONSTANTS
# ==================================================
EMOJI = {

    "Positif": "😊",

    "Netral": "😐",

    "Negatif": "☹️"

}


# ==================================================
# VALIDATION
# ==================================================
def validate_input(text):
    """
    Validate user input before prediction.

    Returns
    -------
    None
        If input is valid.

    str
        Error message if validation fails.
    """

    # Remove leading and trailing spaces
    text = text.strip()

    # Empty input
    if not text:

        return "⚠️ Please enter a review before analyzing."

    # Must contain at least one alphabetic character
    if not any(char.isalpha() for char in text):

        return "⚠️ Review must contain alphabetic characters."

    # Extract words that contain alphabetic characters
    words = []

    for word in text.split():

        cleaned = word.strip(".,!?;:()[]{}\"'`~@#$%^&*-_=+/\\|<>")

        if any(char.isalpha() for char in cleaned):

            words.append(cleaned)

    # Minimum two words
    if len(words) < 2:

        return "⚠️ Please enter at least two words."

    return None


# ==================================================
# ROUTES
# ==================================================
@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    text = ""

    error = None

    if request.method == "POST":

        text = request.form["text"]

        # Validate input
        error = validate_input(text)

        if error is None:

            result = predict_sentiment(text)

            result["emoji"] = EMOJI.get(

                result["label"],

                "🤖"

            )

    return render_template(

        "index.html",

        text=text,

        result=result,

        error=error

    )


# ==================================================
# RUN APPLICATION
# ==================================================
if __name__ == "__main__":

    app.run(
        debug=True
    )