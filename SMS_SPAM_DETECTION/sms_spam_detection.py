#!/usr/bin/env python3
"""
SMS Spam Detection Model with Web Interface
This script implements a machine learning model to classify SMS messages as spam or ham.
Includes a plain web interface with no styling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Flask imports for web interface
from flask import Flask, render_template, request, jsonify
import webbrowser
import threading
import time
import os

# Global variables for web interface
vectorizer = None
classifier = None
model_accuracy = 0

def clean_text(text):
    """Simple text cleaning function"""
    if pd.isna(text):
        return "emptytext"
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords
    stop_words = set([
        'i','me','my','myself','we','our','ours','ourselves','you','your',
        'yours','yourself','yourselves','he','him','his','himself','she','her',
        'hers','herself','it','its','itself','they','them','their','theirs',
        'themselves','what','which','who','whom','this','that','these','those',
        'am','is','are','was','were','be','been','being','have','has','had',
        'having','do','does','did','doing','a','an','the','and','but','if','or',
        'because','as','until','while','of','at','by','for','with','about',
        'against','between','into','through','during','before','after','above',
        'below','to','from','up','down','in','out','on','off','over','under',
        'again','further','then','once','here','there','when','where','why',
        'how','all','any','both','each','few','more','most','other','some',
        'such','no','nor','not','only','own','same','so','than','too','very',
        's','t','can','will','just','don','should','now'
    ])
    
    words = [word for word in words if word not in stop_words]
    return ' '.join(words) if words else "emptytext"

def train_model():
    """Train the SMS spam detection model"""
    global vectorizer, classifier, model_accuracy
    
    print("Loading and training the model...")
    
    # Load the dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    
    # Clean the dataset - keep only first two columns
    df = df.iloc[:, :2]
    df.columns = ['label', 'message']
    
    print(f"Dataset loaded: {len(df)} messages")
    
    # Clean the text
    df['cleaned'] = df['message'].apply(clean_text)
    
    # Convert labels to binary
    df['label_binary'] = (df['label'] == 'spam').astype(int)
    
    # Split the data
    X = df['cleaned']
    y = df['label_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=2500)
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)
    
    # Train model
    classifier = MultinomialNB()
    classifier.fit(X_train_vectors, y_train)
    
    # Calculate accuracy
    y_pred = classifier.predict(X_test_vectors)
    model_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully! Accuracy: {model_accuracy:.4f}")

def predict_spam(message):
    """Predict if a message is spam"""
    if classifier is None or vectorizer is None:
        return {"error": "Model not trained yet"}
    
    # Clean the message
    cleaned_message = clean_text(message)
    
    # Transform using the same vectorizer
    message_vector = vectorizer.transform([cleaned_message])
    
    # Make prediction
    prediction = classifier.predict(message_vector)[0]
    probability = classifier.predict_proba(message_vector)[0]
    
    result = {
        'prediction': 'SPAM' if prediction == 1 else 'HAM',
        'confidence': float(max(probability)),
        'spam_probability': float(probability[1]),
        'ham_probability': float(probability[0])
    }
    return result

def main():
    print("SMS Spam Detection Model")
    print("=" * 50)
    
    # 1. Load and explore the data
    print("\n1. Loading and exploring the data...")
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        print(f"Dataset Shape: {df.shape}")
        
        # Clean the dataset
        df = df.iloc[:, :2]  # Keep only first two columns
        df.columns = ['label', 'message']
        
        print(f"Cleaned dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Data preprocessing
    print("\n2. Preprocessing data...")
    df['label_binary'] = (df['label'] == 'spam').astype(int)
    df['message_length'] = df['message'].str.len()
    
    # Split the data
    X = df['message']
    y = df['label_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # 3. Feature extraction
    print("\n3. Extracting features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix shape (training): {X_train_tfidf.shape}")
    print(f"TF-IDF matrix shape (testing): {X_test_tfidf.shape}")
    
    # 4. Model training
    print("\n4. Training models...")
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # 5. Model evaluation
    print("\n5. Model evaluation...")
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"Best Model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")
    
    # 6. Feature importance analysis
    print("\n6. Feature importance analysis...")
    lr_model = results['Logistic Regression']['model']
    feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_importance = lr_model.coef_[0]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    importance_df['abs_importance'] = abs(importance_df['importance'])
    importance_df = importance_df.sort_values('abs_importance', ascending=False)
    
    print("Top 10 features that indicate SPAM:")
    print(importance_df[importance_df['importance'] > 0].head(10)[['feature', 'importance']])
    
    print("\nTop 10 features that indicate HAM:")
    print(importance_df[importance_df['importance'] < 0].head(10)[['feature', 'importance']])
    
    # 7. Interactive prediction function
    def predict_spam(message, model_name='Logistic Regression'):
        """Predict whether a message is spam or ham"""
        message_tfidf = tfidf_vectorizer.transform([message])
        model = results[model_name]['model']
        prediction = model.predict(message_tfidf)[0]
        probability = model.predict_proba(message_tfidf)[0]
        
        result = {
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'confidence': max(probability),
            'spam_probability': probability[1],
            'ham_probability': probability[0]
        }
        return result
    
    # 8. Test predictions
    print("\n7. Testing predictions...")
    test_messages = [
        "Hello, how are you?",
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
        "Can you call me later?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"
    ]
    
    for i, message in enumerate(test_messages, 1):
        result = predict_spam(message)
        print(f"\nMessage {i}: {message}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Spam Probability: {result['spam_probability']:.3f}")
        print(f"Ham Probability: {result['ham_probability']:.3f}")
    
    # 9. Summary
    print("\n" + "=" * 50)
    print("SMS Spam Detection Model Summary")
    print("=" * 50)
    print(f"Dataset Size: {len(df)} messages")
    print(f"Spam Messages: {len(df[df['label'] == 'spam'])} ({len(df[df['label'] == 'spam'])/len(df)*100:.1f}%)")
    print(f"Ham Messages: {len(df[df['label'] == 'ham'])} ({len(df[df['label'] == 'ham'])/len(df)*100:.1f}%)")
    
    print("\nModel Performance:")
    for name, result in results.items():
        print(f"{name}: {result['accuracy']:.4f} accuracy")
    
    print(f"\nBest Model: {best_model[0]} with {best_model[1]['accuracy']:.4f} accuracy")
    
    print("\nKey Features for Spam Detection:")
    print("• Words like 'free', 'win', 'prize', 'urgent', 'call now'")
    print("• Numbers and special characters")
    print("• Promotional language")
    print("• Urgency indicators")
    
    print("\nKey Features for Ham Detection:")
    print("• Personal language")
    print("• Common greetings and conversational words")
    print("• Family and friend references")
    print("• Natural conversation patterns")
    
    print("\nModel successfully trained and evaluated!")

# Flask app for web interface
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'})
    
    result = predict_spam(message)
    return jsonify(result)

def create_templates():
    """Create the HTML template with absolutely no styling"""
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>SMS Spam Detector</title>
</head>
<body>
    <form id="spamForm">
        <input type="text" id="message" placeholder="Enter message..." required>
        <br><br>
        <button type="submit">Detect</button>
    </form>
    
    <div id="result" style="display: none;">
        <p id="predictionText"></p>
    </div>

    <script>
        document.getElementById('spamForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const message = document.getElementById('message').value;
            if (!message.trim()) {
                alert('Please enter a message');
                return;
            }
            
            // Show loading
            const result = document.getElementById('result');
            result.style.display = 'block';
            document.getElementById('predictionText').textContent = 'Analyzing...';
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('predictionText').textContent = 'Error: ' + data.error;
                    return;
                }
                
                const isSpam = data.prediction === 'SPAM';
                
                // Update result display
                document.getElementById('predictionText').textContent = 
                    isSpam ? 'SPAM DETECTED!' : 'LEGITIMATE MESSAGE';
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('predictionText').textContent = 'Error: Could not analyze message';
            });
        });
    </script>
</body>
</html>
    '''
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Write the HTML file
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:5000')

def run_web_interface():
    """Run the web interface"""
    print("Starting SMS Spam Detector Web Interface...")
    print("=" * 40)
    
    # Train the model
    train_model()
    
    # Create HTML template
    create_templates()
    
    print("Model trained successfully!")
    print("Starting web server...")
    print("Opening browser automatically...")
    print("=" * 40)
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # Run the web interface
    run_web_interface()
    
    # Uncomment the line below to run the original analysis instead
    # main() 