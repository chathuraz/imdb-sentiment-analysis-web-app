import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK data if not present
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)


BASE_DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_DIR, 'Train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'Test.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_model.joblib')


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    # Load training data
    train_df = pd.read_csv(train_path)
    train_df = train_df.dropna(subset=['text', 'label'])
    X_train = train_df['text'].apply(preprocess_text).values
    y_train = train_df['label'].astype(int).values
    
    # Load test data
    test_df = pd.read_csv(test_path)
    test_df = test_df.dropna(subset=['text', 'label'])
    X_test = test_df['text'].apply(preprocess_text).values
    y_test = test_df['label'].astype(int).values
    
    return X_train, X_test, y_train, y_test


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    print('Training model...')
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Accuracy on test set: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f'Model saved to: {MODEL_PATH}')


if __name__ == '__main__':
    main()
