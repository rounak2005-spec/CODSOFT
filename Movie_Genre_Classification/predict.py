import pickle
from data_loader import load_data
from utils import clean_text, decode_genres
import numpy as np

# Load test data
test_data = load_data('test_data.txt', has_genre=False)
titles, plots = zip(*test_data)
cleaned_plots = [clean_text(p) for p in plots]

# Load models
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Vectorize
X_test = vectorizer.transform(cleaned_plots)

# Predict with lower threshold for better genre detection
Y_pred_proba = model.predict_proba(X_test)
# Use threshold of 0.1 instead of default 0.5
threshold = 0.1
Y_pred = (Y_pred_proba > threshold).astype(int)

predicted_genres = decode_genres(Y_pred, mlb)

# Output predictions
for idx, (title, genres) in enumerate(zip(titles, predicted_genres), 1):
    print(f"{idx} ::: {title} ::: {', '.join(genres)}")
