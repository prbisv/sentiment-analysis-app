import re
import csv
import json
import unicodedata
from pathlib import Path

from nltk.tokenize import TweetTokenizer

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==================================================
# PATH CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "resources"

# ==================================================
# LOAD SLANG DICTIONARY
# ==================================================
SLANG_DICT = {}

with open(
    RESOURCE_DIR / "slang_word_update.csv",
    encoding="utf-8"
) as f:

    reader = csv.reader(f)

    for key, value in reader:
        SLANG_DICT[key] = value

if SLANG_DICT:
    SLANG_PATTERN = re.compile(
        r"\b(" + "|".join(map(re.escape, SLANG_DICT.keys())) + r")\b"
    )
else:
    SLANG_PATTERN = None

# ==================================================
# LOAD KBBI
# ==================================================
with open(
    RESOURCE_DIR / "dataKBBI.json",
    encoding="utf-8"
) as f:

    KBBI = set(json.load(f))

tweet_tokenizer = TweetTokenizer(
    strip_handles=True,
    reduce_len=True
)

# ==================================================
# LOAD STOPWORDS
# ==================================================
stopword_factory = StopWordRemoverFactory()

STOPWORDS = set(
    stopword_factory.get_stop_words()
)

# ==================================================
# LOAD STEMMER
# ==================================================
stemmer_factory = StemmerFactory()

STEMMER = stemmer_factory.create_stemmer()

# ==================================================
# PREPROCESSING FUNCTIONS
# ==================================================
def lower_text(text):
    return text.lower()


def remove_character(text):

    text = unicodedata.normalize(
        "NFKD",
        text
    ).encode(
        "ASCII",
        "ignore"
    ).decode("utf-8")

    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'(\w)(\1{2,})', r'\1', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r',', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def normalize_slang(text):

    if SLANG_PATTERN is None:
        return text

    return SLANG_PATTERN.sub(
        lambda match: SLANG_DICT[match.group()],
        text
    )


def cek_kbbi(text):

    words = tweet_tokenizer.tokenize(text)

    words = [
        word
        for word in words
        if word in KBBI
    ]

    return " ".join(words)


def remove_stopwords(text):

    words = text.split()

    words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    return " ".join(words)


def stemming(text):

    return STEMMER.stem(text)


# ==================================================
# MAIN PREPROCESS FUNCTION
# ==================================================
def preprocess(text):

    text = lower_text(text)
    text = remove_character(text)
    text = normalize_slang(text)
    text = cek_kbbi(text)
    text = remove_stopwords(text)
    text = stemming(text)

    return text.strip()


# ==================================================
# DEBUG FUNCTION
# ==================================================
def preprocess_steps(text):

    result = {}

    result["lower"] = lower_text(text)

    result["remove_character"] = remove_character(
        result["lower"]
    )

    result["normalize_slang"] = normalize_slang(
        result["remove_character"]
    )

    result["kbbi"] = cek_kbbi(
        result["normalize_slang"]
    )

    result["stopword"] = remove_stopwords(
        result["kbbi"]
    )

    result["stemming"] = stemming(
        result["stopword"]
    )

    return result