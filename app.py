from flask import Flask, request, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Download NLTK data (run once if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer at startup
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize preprocessing components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define preprocessing function (same as in notebook)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Handle common contractions: don't/doesn't -> do not
    text = text.replace("n't", " not")

    # Remove quotes and some punctuation we don't need (keep word characters and spaces)
    text = re.sub(r"[\"'`,:;()\-]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Define negation words to preserve
    negations = {"not", "no", "never", "none"}

    # Create a local stopword set that keeps negations
    local_stopwords = stop_words - negations

    words = text.split()

    # Simple negation handling: convert the word immediately following a negation into a NOT_<word> token
    processed_words = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in negations and i + 1 < len(words):
            # keep the negation token (optional) and add a NOT_ prefixed token for the next word
            next_w = words[i + 1]
            # skip stopwords for the next word but still mark negation
            negated = "NOT_" + lemmatizer.lemmatize(next_w)
            processed_words.append(w)
            processed_words.append(negated)
            i += 2
        else:
            if w not in local_stopwords:
                processed_words.append(lemmatizer.lemmatize(w))
            i += 1

    return ' '.join(processed_words)

# Home route - display form and handle predictions
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get text from form
        review_text = request.form.get('review', '')

        if not review_text.strip():
            return render_template('index.html', error='Please enter some text to analyze.')

        # Preprocess the text
        processed_text = preprocess_text(review_text)

        # Vectorize the text
        text_vectorized = vectorizer.transform([processed_text])

        # Make prediction
        prediction = model.predict(text_vectorized)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = max(probabilities)

        return render_template('index.html', prediction=prediction, confidence=confidence, text=review_text)
    
    return render_template('index.html')

# Analytics route - display dataset statistics and chart
@app.route('/analytics')
def analytics():
    # Load the training dataset
    df = pd.read_csv('Train.csv')
    
    # Compute statistics
    positive_count = (df['label'] == 1).sum()
    negative_count = (df['label'] == 0).sum()
    total_reviews = len(df)
    
    # Generate bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['Negative', 'Positive'], [negative_count, positive_count], color=['#e74c3c', '#2ecc71'])
    ax.set_title('Sentiment Distribution in Training Dataset', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Reviews', fontsize=12)
    ax.set_xlabel('Sentiment', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate([negative_count, positive_count]):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Save chart to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return render_template('analytics.html', 
                         positive_count=positive_count, 
                         negative_count=negative_count, 
                         total_reviews=total_reviews, 
                         chart=image_base64)

if __name__ == '__main__':
    app.run(debug=True)
