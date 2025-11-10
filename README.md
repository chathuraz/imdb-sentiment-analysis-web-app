# IMDB Sentiment Analysis Web App

A web application that performs sentiment analysis on movie reviews using machine learning. The app analyzes text input to determine whether a review is positive or negative, with special handling for negation (e.g., "not good" correctly predicts negative sentiment).

## Tech Stack

- **Python 3.13** - Core programming language
- **scikit-learn** - Machine learning library for TF-IDF vectorization and LogisticRegression
- **Flask** - Web framework for the application
- **TF-IDF** - Text feature extraction technique
- **NLTK** - Natural language processing for text preprocessing
- **pandas** - Data manipulation and analysis
- **matplotlib** - Chart generation for analytics
- **joblib** - Model serialization

## Features

- **Sentiment Prediction**: Analyze text input and predict positive or negative sentiment with confidence scores
- **Web Interface**: Clean, responsive web UI built with Flask and Jinja2 templates
- **Analytics Page**: View dataset statistics and interactive bar chart showing sentiment distribution
- **Negation Handling**: Advanced text preprocessing that correctly handles negations (e.g., "not bad" → negative)
- **Model Performance**: 89% accuracy on test data with balanced performance across classes

## Usage

### Web Interface

1. **Main Sentiment Analysis Page** (`http://127.0.0.1:5000`):
   - Enter any text (movie review, product feedback, etc.) in the text area
   - Click "Analyze Sentiment" to get instant results
   - View the predicted sentiment (Positive/Negative) with confidence percentage
   - See the original text displayed for reference

2. **Analytics Dashboard** (`http://127.0.0.1:5000/analytics`):
   - View overall dataset statistics (total reviews, positive/negative counts)
   - Interactive bar chart showing sentiment distribution
   - Navigate back to the main analysis page

### What It Provides

- **Instant Sentiment Analysis**: Real-time prediction of text sentiment with probability scores
- **Educational Tool**: Learn about natural language processing and machine learning through a practical web application
- **Data Insights**: Understand sentiment patterns in text data through the analytics dashboard
- **Negation-Aware Processing**: Accurately handles complex language constructs like negations
- **High Accuracy**: 89% accuracy model trained on large IMDB dataset for reliable predictions
- **Web-Based Accessibility**: No installation required for end-users, accessible via any web browser

### Example Usage

- **Movie Reviews**: "This film was absolutely fantastic!" → Positive (95% confidence)
- **Product Feedback**: "The product works well but not great." → Negative (78% confidence)
- **Social Media**: "Not disappointed with the service." → Positive (82% confidence)

### Prerequisites
- Python 3.8 or higher
- Git (optional, for cloning)

### Quick Start

1. **Clone or navigate to the project directory**
   ```bash
   cd F:\projects\AI
   ```

2. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```powershell
   python app.py
   ```

5. **Open your browser**
   - Main app: http://127.0.0.1:5000
   - Analytics: http://127.0.0.1:5000/analytics

## How It Works

### ML Pipeline Overview

1. **Data Loading**: The model was trained on the IMDB movie reviews dataset (40,000 reviews)

2. **Text Preprocessing**:
   - Convert to lowercase
   - Handle contractions (don't → do not)
   - Remove punctuation and normalize whitespace
   - Apply lemmatization using WordNet
   - **Negation Handling**: Preserve negation words and mark following words (e.g., "not good" becomes "not NOT_good")

3. **Feature Extraction**:
   - TF-IDF vectorization with unigrams and bigrams
   - Limited to top 10,000 features for efficiency

4. **Model Training**:
   - LogisticRegression with hyperparameter tuning via GridSearchCV
   - Optimized parameters: C=1.0, solver='lbfgs'
   - 5-fold cross-validation for robust evaluation

5. **Prediction**:
   - Input text is preprocessed using the same pipeline
   - Transformed to TF-IDF features
   - Model predicts sentiment (0=negative, 1=positive)
   - Returns prediction with confidence probability

### Model Performance
- **Accuracy**: 89.05%
- **F1-Score**: 89.05%
- **Cross-validation F1**: 89.18%
- Balanced performance across positive and negative classes

## Future Improvements

1. **Model Enhancement**: Implement ensemble methods (Random Forest, XGBoost) or deep learning approaches (LSTM, BERT) for improved accuracy

2. **Real-time Feedback Loop**: Add user feedback mechanism to collect corrections and implement active learning for model retraining

3. **API Development**: Create REST API endpoints for programmatic access to sentiment analysis functionality

4. **Batch Processing**: Support for analyzing multiple reviews at once with CSV upload/download capabilities

5. **Advanced Analytics**: Add more detailed analytics including word clouds, temporal trends, and model performance monitoring

6. **Deployment**: Containerize with Docker and deploy to cloud platforms (Azure App Service, AWS Elastic Beanstalk)

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is for educational purposes. Please ensure compliance with IMDB dataset usage terms.
