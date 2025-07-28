# SMS Spam Detector Web Application

A simple web-based SMS spam detection tool that opens in your browser when you run a command in the terminal.

## 🚀 Quick Start

### Option 1: Command Line
```bash
python spam_detector_web.py
```

### Option 2: Double-click (Windows)
Double-click `start_spam_detector.bat`

## 📱 What Happens

1. **Model Training**: The app will train the machine learning model (takes a few seconds)
2. **Browser Opens**: Your default browser will automatically open to `http://localhost:5000`
3. **Web Interface**: You'll see a clean interface with:
   - A text area to enter your message
   - A "Detect Spam" button
   - Example buttons to test different messages
4. **Results**: Get instant results showing if the message is spam or legitimate

## 🎯 Features

- **Simple Interface**: Clean, modern web design
- **Real-time Detection**: Instant results with confidence scores
- **Example Messages**: Click buttons to test different types of messages
- **Visual Feedback**: Green for legitimate, red for spam
- **Probability Display**: Shows spam vs ham probability percentages

## 📝 How to Use

1. **Enter a message** in the text area
2. **Click "🔍 Detect Spam"** button
3. **View results**:
   - 🚨 SPAM DETECTED! (red background)
   - ✅ LEGITIMATE MESSAGE (green background)
   - Confidence bar and probability percentages

## 🧪 Test Examples

The web interface includes example buttons for testing:

**Legitimate Messages:**
- "Hello, how are you?"
- "Can you call me later?"

**Spam Messages:**
- "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121"
- "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"

## 🔧 Technical Details

- **Machine Learning**: Naive Bayes classifier with TF-IDF features
- **Text Processing**: Custom text cleaning and stopword removal
- **Accuracy**: ~97% on test data
- **Training Data**: 5,572 SMS messages (747 spam, 4,825 legitimate)

## 📁 Files

- `spam_detector_web.py` - Main web application
- `start_spam_detector.bat` - Windows batch file for easy startup
- `spam.csv` - Training dataset
- `requirements.txt` - Python dependencies

## 🛠️ Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - flask

## 📦 Installation

1. Make sure you have Python installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the web app:
   ```bash
   python spam_detector_web.py
   ```

## 🎉 Enjoy!

The web application will automatically open in your browser where you can start testing SMS spam detection immediately! 