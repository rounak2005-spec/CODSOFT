import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from data_loader import load_data
from utils import clean_text, encode_genres

data = load_data('train_data.txt')

titles, genres, plots = zip(*data)
cleaned_plots = [clean_text(p) for p in plots]

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(cleaned_plots)
Y_train, mlb = encode_genres(genres)

model = OneVsRestClassifier(LogisticRegression(max_iter=500))
model.fit(X_train, Y_train)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)
