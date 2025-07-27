import re
from sklearn.preprocessing import MultiLabelBinarizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def encode_genres(genres_list):
    mlb = MultiLabelBinarizer()
    binary_genres = mlb.fit_transform(genres_list)
    return binary_genres, mlb

def decode_genres(binary_genres, mlb):
    return mlb.inverse_transform(binary_genres)
