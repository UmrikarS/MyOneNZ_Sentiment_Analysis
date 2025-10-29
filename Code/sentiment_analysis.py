"""
One NZ Mobile App Review Sentiment Analysis
Advanced Business Analytics Project with AI/ML Techniques

Author: [Sneha]
Date: October 2025
Course: MBI806B - Business Data Analytics
"""

# ============================================================================
# SECTION 1: SETUP AND IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import re
import time
# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from scipy import stats

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Sentiment Analysis
from nltk.sentiment import SentimentIntensityAnalyzer

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

# Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Visualization
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("All libraries imported successfully!")
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# SECTION 2: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploration
    """
    print("\n" + "=" * 80)
    print("SECTION 2: DATA LOADING AND EXPLORATION")
    print("=" * 80)

    # Load data
    df = pd.read_csv(file_path)

    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Reviews: {len(df):,}")

    print("\nColumn Names and Types:")
    print(df.dtypes)

    print("\nFirst Few Records:")
    print(df.head(3))

    print("\nBasic Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])

    return df


# ============================================================================
# SECTION 3: DATA PREPROCESSING AND CLEANING
# ============================================================================

def preprocess_data(df):
    """
    Clean and preprocess the dataset with advanced NLP techniques
    """
    print("\n" + "=" * 80)
    print("SECTION 3: DATA PREPROCESSING WITH ADVANCED NLP")
    print("=" * 80)

    df_clean = df.copy()

    # 1. Convert date columns
    print("\n1. Processing date columns...")
    df_clean['at'] = pd.to_datetime(df_clean['at'], format='%m/%d/%Y %H:%M', errors='coerce')
    df_clean['repliedAt'] = pd.to_datetime(df_clean['repliedAt'], format='%m/%d/%Y %H:%M', errors='coerce')

    # 2. Extract temporal features
    print("2. Extracting temporal features...")
    df_clean['year'] = df_clean['at'].dt.year
    df_clean['month'] = df_clean['at'].dt.month
    df_clean['quarter'] = df_clean['at'].dt.quarter
    df_clean['day_of_week'] = df_clean['at'].dt.day_name()
    df_clean['review_date'] = df_clean['at'].dt.date

    # 3. Create binary features
    print("3. Creating binary features...")
    df_clean['has_reply'] = df_clean['replyContent'].notna().astype(int)
    df_clean['has_version'] = df_clean['appVersion'].notna().astype(int)

    # 4. Calculate response time
    print("4. Calculating response time...")
    df_clean['response_time_hours'] = (
            (df_clean['repliedAt'] - df_clean['at']).dt.total_seconds() / 3600
    )

    # 5. Text preprocessing - Multiple versions for different purposes
    print("5. Preprocessing review text...")
    print("   a. Basic cleaning (for sentiment analysis)...")
    df_clean['content_clean'] = df_clean['content'].apply(
        lambda x: clean_text(x, remove_stopwords=False, apply_lemmatization=False)
    )

    print("   b. Stopword removal (for word clouds and topic modeling)...")
    df_clean['content_clean_nostop'] = df_clean['content'].apply(
        lambda x: clean_text(x, remove_stopwords=True, apply_lemmatization=False)
    )

    print("   c. Lemmatization (for semantic analysis)...")
    df_clean['content_lemmatized'] = df_clean['content'].apply(
        lambda x: clean_text(x, remove_stopwords=True, apply_lemmatization=True)
    )

    # 6. Text metrics
    print("6. Calculating text metrics...")
    df_clean['review_length'] = df_clean['content_clean'].apply(len)
    df_clean['word_count'] = df_clean['content_clean'].apply(lambda x: len(x.split()))
    df_clean['word_count_nostop'] = df_clean['content_clean_nostop'].apply(lambda x: len(x.split()))

    # 7. Version grouping
    print("7. Grouping app versions...")
    df_clean['major_version'] = df_clean['appVersion'].apply(extract_major_version)

    # 8. Sentiment category based on score
    print("8. Creating sentiment labels...")
    df_clean['sentiment_label'] = df_clean['score'].apply(categorize_sentiment)

    print(f"\nCleaned dataset shape: {df_clean.shape}")
    print(f"\nText processing summary:")
    print(f"  - Average words (original): {df_clean['word_count'].mean():.1f}")
    print(f"  - Average words (no stopwords): {df_clean['word_count_nostop'].mean():.1f}")
    print(
        f"  - Stopwords removed: {((df_clean['word_count'].mean() - df_clean['word_count_nostop'].mean()) / df_clean['word_count'].mean() * 100):.1f}%")
    print("\nPreprocessing complete!")

    return df_clean


def clean_text(text, remove_stopwords=False, apply_lemmatization=False):
    """
    Advanced text preprocessing with multiple NLP techniques

    Parameters:
    -----------
    text : str
        Input text to clean
    remove_stopwords : bool
        Whether to remove stopwords (for word clouds, topic modeling)
    apply_lemmatization : bool
        Whether to apply lemmatization (for semantic analysis)
    """
    if pd.isna(text):
        return ""

    # 1. NORMALIZATION: Convert to lowercase
    text = str(text).lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # 4. Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

    # 5. Remove special characters but keep basic punctuation for sentiment
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)

    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. TOKENIZATION (if stopword removal or lemmatization needed)
    if remove_stopwords or apply_lemmatization:
        tokens = word_tokenize(text)

        # 8. STOPWORD REMOVAL
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            # Add custom stopwords specific to app reviews
            custom_stops = {'app', 'nz', 'one', 'vodafone', 'would', 'could',
                            'get', 'im', 'ive', 'dont', 'cant', 'wont'}
            stop_words.update(custom_stops)
            tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

        # 9. LEMMATIZATION
        if apply_lemmatization:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Reconstruct text
        text = ' '.join(tokens)

    return text


def extract_major_version(version):
    """
    Extract major version number from app version
    """
    if pd.isna(version):
        return 'Unknown'

    match = re.match(r'(\d+)', str(version))
    if match:
        return f"v{match.group(1)}"
    return 'Unknown'


def categorize_sentiment(score):
    """
    Categorize review score into sentiment labels
    """
    if score >= 4:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'


# ============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(df):
    """
    Comprehensive exploratory data analysis
    """
    print("\n" + "=" * 80)
    print("SECTION 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # 1. Score Distribution
    print("\n1. Review Score Distribution:")
    print(df['score'].value_counts().sort_index())
    print(f"\nAverage Score: {df['score'].mean():.2f}")
    print(f"Median Score: {df['score'].median():.2f}")

    # 2. Sentiment Distribution
    print("\n2. Sentiment Distribution:")
    sentiment_counts = df['sentiment_label'].value_counts()
    print(sentiment_counts)
    print(f"\nNegative Review Percentage: {(sentiment_counts['Negative'] / len(df)) * 100:.1f}%")

    # 3. Temporal Analysis
    print("\n3. Reviews Over Time:")
    df['review_date'] = pd.to_datetime(df['review_date'])
    monthly_reviews = df.groupby(df['review_date'].apply(lambda x: x.strftime('%Y-%m'))).size()
    print(f"Peak Month: {monthly_reviews.idxmax()} ({monthly_reviews.max()} reviews)")
    print(f"Lowest Month: {monthly_reviews.idxmin()} ({monthly_reviews.min()} reviews)")

    # 4. App Version Analysis
    print("\n4. Top App Versions:")
    print(df['appVersion'].value_counts().head(10))

    # 5. Company Response Analysis
    print("\n5. Company Response Rate:")
    response_rate = (df['has_reply'].sum() / len(df)) * 100
    print(f"Overall Response Rate: {response_rate:.1f}%")

    avg_response_time = df['response_time_hours'].mean()
    print(f"Average Response Time: {avg_response_time:.1f} hours ({avg_response_time / 24:.1f} days)")

    # 6. Review Length Analysis
    print("\n6. Review Length Statistics:")
    print(f"Average Word Count: {df['word_count'].mean():.1f}")
    print(f"Average Character Length: {df['review_length'].mean():.1f}")

    return df


# ============================================================================
# SECTION 5: ADVANCED SENTIMENT ANALYSIS
# ============================================================================

def perform_sentiment_analysis(df):
    """
    Perform multiple sentiment analysis techniques including deep learning
    """
    print("\n" + "=" * 80)
    print("SECTION 5: ADVANCED SENTIMENT ANALYSIS")
    print("=" * 80)

    # 1. VADER Sentiment Analysis
    print("\n1. VADER Sentiment Analysis...")
    sia = SentimentIntensityAnalyzer()

    df['vader_compound'] = df['content_clean'].apply(
        lambda x: sia.polarity_scores(x)['compound'] if x else 0
    )
    df['vader_sentiment'] = df['vader_compound'].apply(classify_vader_sentiment)

    # 2. TextBlob Sentiment Analysis
    print("2. TextBlob Sentiment Analysis...")
    df['textblob_polarity'] = df['content_clean'].apply(
        lambda x: TextBlob(x).sentiment.polarity if x else 0
    )
    df['textblob_subjectivity'] = df['content_clean'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity if x else 0
    )
    df['textblob_sentiment'] = df['textblob_polarity'].apply(classify_textblob_sentiment)

    # 3. RoBERTa Sentiment Analysis (Deep Learning)
    print("\n3. RoBERTa/BERT Sentiment Analysis...")
    print("   This may take several minutes for large datasets...")

    try:
        df = perform_roberta_sentiment(df)
        roberta_available = True
    except Exception as e:
        print(f"   RoBERTa analysis skipped: {str(e)}")
        print("   To enable, install: pip install transformers torch")
        roberta_available = False
        df['roberta_sentiment'] = 'Not Available'
        df['roberta_score'] = 0

    # 4. Compare with actual ratings
    print("\n4. Sentiment Validation:")
    vader_accuracy = (df['vader_sentiment'] == df['sentiment_label']).mean()
    textblob_accuracy = (df['textblob_sentiment'] == df['sentiment_label']).mean()

    print(f"VADER Accuracy vs Actual Ratings: {vader_accuracy * 100:.2f}%")
    print(f"TextBlob Accuracy vs Actual Ratings: {textblob_accuracy * 100:.2f}%")

    if roberta_available:
        roberta_accuracy = (df['roberta_sentiment'] == df['sentiment_label']).mean()
        print(f"RoBERTa Accuracy vs Actual Ratings: {roberta_accuracy * 100:.2f}%")

    # 5. Sentiment Distribution
    print("\n5. Sentiment Analysis Results:")
    print("\nVADER Sentiment Distribution:")
    print(df['vader_sentiment'].value_counts())
    print("\nTextBlob Sentiment Distribution:")
    print(df['textblob_sentiment'].value_counts())

    if roberta_available:
        print("\nRoBERTa Sentiment Distribution:")
        print(df['roberta_sentiment'].value_counts())

    return df


def perform_roberta_sentiment(df, batch_size=32, max_samples=None):
    """
    Perform sentiment analysis using RoBERTa pre-trained model
    """
    try:
        from transformers import pipeline
        import torch

        # Check if GPU is available
        device = 0 if torch.cuda.is_available() else -1

        # Load pre-trained sentiment analysis pipeline
        # Using cardiffnlp/twitter-roberta-base-sentiment-latest (optimized for social media)
        print("   Loading RoBERTa model...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device,
            truncation=True,
            max_length=512
        )

        # Prepare data
        df_subset = df[df['content_clean'].notna()].copy()

        # Limit samples for demonstration (remove this for full analysis)
        if max_samples and len(df_subset) > max_samples:
            print(f"   Processing {max_samples} samples for demonstration...")
            df_subset = df_subset.sample(n=max_samples, random_state=42)
        else:
            print(f"   Processing {len(df_subset)} reviews...")

        reviews = df_subset['content_clean'].tolist()

        # Process in batches to manage memory
        roberta_results = []
        total_batches = (len(reviews) + batch_size - 1) // batch_size

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            batch_num = i // batch_size + 1

            if batch_num % 10 == 0:
                print(f"   Processing batch {batch_num}/{total_batches}...")

            try:
                results = sentiment_pipeline(batch)
                roberta_results.extend(results)
            except Exception as e:
                print(f"   Error in batch {batch_num}: {str(e)}")
                # Fill with neutral for failed batches
                roberta_results.extend([{'label': 'neutral', 'score': 0.5}] * len(batch))

        # Map labels to our categories
        label_mapping = {
            'positive': 'Positive',
            'negative': 'Negative',
            'neutral': 'Neutral',
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative',
            'NEUTRAL': 'Neutral'
        }

        df_subset['roberta_sentiment'] = [
            label_mapping.get(r['label'], 'Neutral') for r in roberta_results
        ]
        df_subset['roberta_score'] = [r['score'] for r in roberta_results]

        # Merge back to original dataframe
        df = df.merge(
            df_subset[['reviewId', 'roberta_sentiment', 'roberta_score']],
            on='reviewId',
            how='left'
        )

        # Fill NaN values for reviews that weren't processed
        df['roberta_sentiment'] = df['roberta_sentiment'].fillna('Not Processed')
        df['roberta_score'] = df['roberta_score'].fillna(0)

        print("   RoBERTa analysis complete!")

    except ImportError:
        raise Exception("Transformers library not installed")

    return df


def classify_vader_sentiment(compound_score):
    """
    Classify VADER compound score into sentiment categories
    """
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def classify_textblob_sentiment(polarity):
    """
    Classify TextBlob polarity into sentiment categories
    """
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


# ============================================================================
# SECTION 6: TOPIC MODELING AND THEME EXTRACTION
# ============================================================================

def perform_topic_modeling(df, n_topics=5, n_words=10):
    """
    Extract main themes using LDA and NMF with proper preprocessing
    """
    print("\n" + "=" * 80)
    print("SECTION 6: TOPIC MODELING")
    print("=" * 80)

    # Prepare data - use stopword-removed text
    reviews = df['content_lemmatized'].dropna()

    print(f"\nProcessing {len(reviews)} reviews for topic modeling...")

    # Create document-term matrix
    print("\n1. Creating TF-IDF matrix...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=5,  # Ignore terms appearing in less than 5 documents
        max_df=0.7,  # Ignore terms appearing in more than 70% of documents
        ngram_range=(1, 2)  # Include bigrams
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

    print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")

    # LDA Topic Modeling
    print(f"\n2. Performing LDA with {n_topics} topics...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='online'
    )
    lda_model.fit(tfidf_matrix)

    print("\nLDA Topics:")
    display_topics(lda_model, tfidf_vectorizer.get_feature_names_out(), n_words)

    # NMF Topic Modeling
    print(f"\n3. Performing NMF with {n_topics} topics...")
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        max_iter=200,
        init='nndsvda'
    )
    nmf_model.fit(tfidf_matrix)

    print("\nNMF Topics:")
    display_topics(nmf_model, tfidf_vectorizer.get_feature_names_out(), n_words)

    # Get topic distribution for each document
    print("\n4. Assigning topics to reviews...")
    lda_topics = lda_model.transform(tfidf_matrix)
    df.loc[reviews.index, 'dominant_topic'] = lda_topics.argmax(axis=1)

    print("\nTopic Distribution:")
    print(df['dominant_topic'].value_counts().sort_index())

    return lda_model, nmf_model, tfidf_matrix, tfidf_vectorizer


def display_topics(model, feature_names, n_words):
    """
    Display top words for each topic
    """
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")


# ============================================================================
# SECTION 7: MACHINE LEARNING CLASSIFICATION
# ============================================================================

def build_ml_models(df):
    """
    Build and evaluate multiple ML classification models
    """
    print("\n" + "=" * 80)
    print("SECTION 7: MACHINE LEARNING MODELS")
    print("=" * 80)

    # Prepare data
    print("\n1. Preparing training data...")
    df_ml = df[df['content_clean'].notna()].copy()

    X = df_ml['content_clean']
    y = df_ml['sentiment_label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Vectorization
    print("\n2. Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42, probability=True),
        'Naive Bayes': MultinomialNB(),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate models
    print("\n3. Training and evaluating models...")
    results = {}

    for name, model in models.items():
        print(f"\n   Training {name}...")
        start_time = time.time()
        model.fit(X_train_vec, y_train)
        training_time = time.time() - start_time
        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        print(f"   Cross-validating {name}...")
        # 5-fold cross-validation
        cv_scores = cross_val_score(
            model,
            X_train_vec,
            y_train,
            cv=5,
            scoring='f1_weighted',  # Use F1 for multi-class
            n_jobs=-1  # Use all CPU cores for speed
        )

        results[name] = {
            'model': model,
            'training_time': training_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'y_test': y_test,  # Save for confusion matrix
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"Training completed in {training_time:.2f} seconds.")

    # Best model
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\n4. Best Model: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")

    # Detailed classification report for best model
    print(f"\n5. Detailed Classification Report for {best_model_name}:")
    print(classification_report(y_test, results[best_model_name]['predictions']))

    # Return results with y_test for visualization
    return results, vectorizer, y_test


# ============================================================================
# SECTION 7.1: ASSIGN TOPICS TO REVIEWS
# ============================================================================

def assign_dominant_topics(df, lda_model, tfidf_matrix):
    """
    Assign dominant topic to each review based on LDA model

    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset
    lda_model : LatentDirichletAllocation
        Fitted LDA model
    tfidf_matrix : sparse matrix
        Document-term matrix used for training

    Returns:
    --------
    pandas.Series : Dominant topic for each review
    """

    print("\n" + "=" * 80)
    print("SECTION 7.1: ASSIGNING DOMINANT TOPICS TO REVIEWS")
    print("=" * 80)

    # Get topic distributions for all documents
    topic_distributions = lda_model.transform(tfidf_matrix)

    # Get dominant topic (highest probability) for each document
    dominant_topics = np.argmax(topic_distributions, axis=1)

    print(f"\nTotal reviews processed: {len(dominant_topics):,}")

    # Topic distribution across reviews
    topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
    print(f"\nReviews per Topic:")
    for topic_id, count in topic_counts.items():
        pct = (count / len(dominant_topics)) * 100
        print(f"  Topic {topic_id}: {count:6,} reviews ({pct:5.1f}%)")

    return pd.Series(dominant_topics, name='dominant_topic', index=df.index)


# ============================================================================
# SECTION 7.2: CALCULATE AVERAGE RATING PER TOPIC
# ============================================================================

def calculate_average_rating_by_topic(df, dominant_topics):
    """
    Calculate average rating for each topic

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with 'score' column
    dominant_topics : pandas.Series
        Dominant topic per review

    Returns:
    --------
    pandas.DataFrame : Topic statistics
    """

    print("\n" + "=" * 80)
    print("SECTION 7.2: AVERAGE RATING BY TOPIC")
    print("=" * 80)

    # Add dominant topic to dataframe
    df_analysis = df.copy()
    df_analysis['dominant_topic'] = dominant_topics

    # Calculate statistics by topic
    topic_stats = df_analysis.groupby('dominant_topic').agg({
        'score': ['mean', 'median', 'std', 'count', 'min', 'max']
    }).round(3)

    # Flatten column names
    topic_stats.columns = ['Mean_Rating', 'Median_Rating', 'Std_Dev',
                           'Review_Count', 'Min_Rating', 'Max_Rating']

    # Overall average
    overall_avg = df_analysis['score'].mean()

    print(f"\nOverall Average Rating (All Reviews): {overall_avg:.2f}/5.0")
    print(f"\nAverage Rating by Topic:\n")
    print(topic_stats.to_string())

    # Identify best and worst topics
    best_topic = topic_stats['Mean_Rating'].idxmax()
    worst_topic = topic_stats['Mean_Rating'].idxmin()

    print(f"\n\nBest Topic: Topic {best_topic} (Avg: {topic_stats.loc[best_topic, 'Mean_Rating']:.2f})")
    print(f"Worst Topic: Topic {worst_topic} (Avg: {topic_stats.loc[worst_topic, 'Mean_Rating']:.2f})")
    print(
        f"Difference: {topic_stats.loc[best_topic, 'Mean_Rating'] - topic_stats.loc[worst_topic, 'Mean_Rating']:.2f} stars")

    return df_analysis, topic_stats


# ============================================================================
# SECTION 7.3: EXTRACT TOPIC KEYWORDS
# ============================================================================

def extract_topic_keywords(lda_model, vectorizer, n_words=15):
    """
    Extract key terms from each topic

    Parameters:
    -----------
    lda_model : LatentDirichletAllocation
        Fitted LDA model
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    n_words : int
        Number of top words to extract per topic

    Returns:
    --------
    dict : Keywords for each topic
    """

    print("\n" + "=" * 80)
    print("SECTION 7.3: EXTRACTING TOPIC KEYWORDS")
    print("=" * 80)

    feature_names = np.array(vectorizer.get_feature_names_out())
    n_topics = lda_model.n_components

    topic_keywords = {}

    print(f"\nTop {n_words} Words per Topic:\n")

    for topic_id in range(n_topics):
        # Get top word indices
        top_indices = np.argsort(lda_model.components_[topic_id])[-n_words:][::-1]

        # Get words and weights
        keywords = feature_names[top_indices]
        weights = lda_model.components_[topic_id][top_indices]

        topic_keywords[topic_id] = {
            'keywords': list(keywords),
            'weights': list(weights)
        }

        # Display
        print(f"Topic {topic_id}:")
        keyword_str = ", ".join(keywords[:10])
        print(f"  {keyword_str}")
        print()

    return topic_keywords


# ============================================================================
# SECTION 7.4: CALCULATE AVERAGE RATING FOR KEYWORD PRESENCE
# ============================================================================

def calculate_rating_by_keyword_presence(df_analysis, topic_keywords,
                                         text_column='content_clean'):
    """
    Calculate average rating for reviews containing topic keywords

    Parameters:
    -----------
    df_analysis : pandas.DataFrame
        Dataset with scores and text
    topic_keywords : dict
        Keywords for each topic
    text_column : str
        Column with text to search

    Returns:
    --------
    pandas.DataFrame : Statistics for keyword presence
    """

    print("\n" + "=" * 80)
    print("SECTION 7.4: AVERAGE RATING BY KEYWORD PRESENCE")
    print("=" * 80)

    def contains_keywords(text, keywords, min_keywords=1):
        """Check if text contains at least min_keywords"""
        if not isinstance(text, str):
            return False

        count = 0
        for keyword in keywords:
            if keyword in text.lower():
                count += 1
                if count >= min_keywords:
                    return True
        return False

    # Calculate for each topic
    keyword_presence_stats = []

    for topic_id, keywords_data in topic_keywords.items():
        keywords = keywords_data['keywords']

        # Create boolean mask for reviews containing keywords
        contains_mask = df_analysis[text_column].apply(
            lambda x: contains_keywords(x, keywords, min_keywords=1)
        )

        # Reviews with and without keywords
        reviews_with_keywords = df_analysis[contains_mask]
        reviews_without_keywords = df_analysis[~contains_mask]

        # Calculate statistics
        if len(reviews_with_keywords) > 0:
            with_keywords_avg = reviews_with_keywords['score'].mean()
            with_keywords_count = len(reviews_with_keywords)
        else:
            with_keywords_avg = np.nan
            with_keywords_count = 0

        if len(reviews_without_keywords) > 0:
            without_keywords_avg = reviews_without_keywords['score'].mean()
            without_keywords_count = len(reviews_without_keywords)
        else:
            without_keywords_avg = np.nan
            without_keywords_count = 0

        rating_diff = with_keywords_avg - without_keywords_avg

        keyword_presence_stats.append({
            'topic_id': topic_id,
            'keywords': ', '.join(keywords[:5]),
            'with_keywords_avg': with_keywords_avg,
            'with_keywords_count': with_keywords_count,
            'without_keywords_avg': without_keywords_avg,
            'without_keywords_count': without_keywords_count,
            'rating_difference': rating_diff
        })

    stats_df = pd.DataFrame(keyword_presence_stats)

    print(f"\nAverage Rating by Keyword Presence:\n")
    print(stats_df[['topic_id', 'keywords', 'with_keywords_avg',
                    'without_keywords_avg', 'rating_difference']].to_string(index=False))

    return df_analysis, topic_keywords, stats_df


# ============================================================================
# SECTION 7.5: PERFORM STATISTICAL TESTS
# ============================================================================

def perform_statistical_tests(df_analysis, topic_keywords, text_column='content_clean'):
    """
    Perform t-tests and ANOVA to assess statistical significance

    Parameters:
    -----------
    df_analysis : pandas.DataFrame
        Dataset with scores and dominant topics
    topic_keywords : dict
        Keywords for each topic
    text_column : str
        Column with text

    Returns:
    --------
    pandas.DataFrame : Statistical test results
    """

    print("\n" + "=" * 80)
    print("SECTION 7.5: STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)

    overall_avg = df_analysis['score'].mean()
    print(f"\nOverall Average Rating: {overall_avg:.2f}/5.0\n")

    def contains_keywords(text, keywords):
        if not isinstance(text, str):
            return False
        for keyword in keywords:
            if keyword in text.lower():
                return True
        return False

    test_results = []

    print("Independent Samples T-Tests (Topic vs. Other Reviews):\n")
    print("-" * 80)

    for topic_id, keywords_data in topic_keywords.items():
        keywords = keywords_data['keywords']

        # Group 1: Reviews containing topic keywords
        topic_mask = df_analysis[text_column].apply(
            lambda x: contains_keywords(x, keywords)
        )
        topic_scores = df_analysis[topic_mask]['score']
        other_scores = df_analysis[~topic_mask]['score']

        # Only perform test if both groups have sufficient data
        if len(topic_scores) > 1 and len(other_scores) > 1:
            # Perform Welch's t-test (doesn't assume equal variances)
            t_stat, p_value = stats.ttest_ind(topic_scores, other_scores, equal_var=False)

            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(topic_scores) - 1) * topic_scores.std() ** 2 +
                                  (len(other_scores) - 1) * other_scores.std() ** 2) /
                                 (len(topic_scores) + len(other_scores) - 2))
            cohens_d = (topic_scores.mean() - other_scores.mean()) / pooled_std if pooled_std > 0 else 0

            # Determine significance and impact
            is_significant = p_value < 0.05
            if is_significant:
                if topic_scores.mean() < other_scores.mean():
                    impact = "NEGATIVE"
                    impact_desc = "Significantly lower ratings"
                else:
                    impact = "POSITIVE"
                    impact_desc = "Significantly higher ratings"
            else:
                impact = "NOT SIGNIFICANT"
                impact_desc = "No significant difference"

            test_results.append({
                'topic_id': topic_id,
                'keywords': ', '.join(keywords[:5]),
                'topic_avg': topic_scores.mean(),
                'other_avg': other_scores.mean(),
                'topic_n': len(topic_scores),
                'other_n': len(other_scores),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'is_significant': is_significant,
                'impact': impact
            })

            # Print results
            print(f"Topic {topic_id} (Keywords: {', '.join(keywords[:5])}...)")
            print(f"  Reviews with keywords: {len(topic_scores)} (Avg: {topic_scores.mean():.2f})")
            print(f"  Reviews without keywords: {len(other_scores)} (Avg: {other_scores.mean():.2f})")
            print(f"  T-statistic: {t_stat:.4f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Cohen's d: {cohens_d:.4f}")
            print(f"  Result: {impact_desc}")
            print("-" * 80)
        else:
            print(f"Topic {topic_id}: Insufficient data for testing")
            print("-" * 80)

    results_df = pd.DataFrame(test_results)
    return results_df


# ============================================================================
# SECTION 7.6: SUMMARIZE FINDINGS
# ============================================================================

def summarize_impact_findings(test_results):
    """
    Summarize topics with significant positive/negative impact

    Parameters:
    -----------
    test_results : pandas.DataFrame
        Results from statistical tests

    Returns:
    --------
    dict : Organized findings
    """

    print("\n" + "=" * 80)
    print("SECTION 7.6: IMPACT SUMMARY")
    print("=" * 80)

    # Filter significant results
    significant = test_results[test_results['is_significant']].copy()
    significant = significant.sort_values('cohens_d', key=abs, ascending=False)

    # Separate positive and negative impact
    positive_impact = significant[significant['impact'] == 'POSITIVE']
    negative_impact = significant[significant['impact'] == 'NEGATIVE']

    print(f"\n{'POSITIVE IMPACT TOPICS':^80}")
    print("=" * 80)
    print(f"Topics associated with significantly HIGHER user ratings:\n")

    if len(positive_impact) > 0:
        for idx, row in positive_impact.iterrows():
            print(f"Topic {int(row['topic_id'])}: {row['keywords']}")
            print(f"  Average Rating: {row['topic_avg']:.2f}/5.0")
            print(f"  vs. Other Topics: {row['other_avg']:.2f}/5.0")
            print(f"  Difference: +{row['topic_avg'] - row['other_avg']:.2f} stars")
            print(f"  Effect Size (Cohen's d): {row['cohens_d']:.4f}", end="")

            # Interpret effect size
            abs_d = abs(row['cohens_d'])
            if abs_d < 0.2:
                effect_interp = "(negligible)"
            elif abs_d < 0.5:
                effect_interp = "(small)"
            elif abs_d < 0.8:
                effect_interp = "(medium)"
            else:
                effect_interp = "(large)"

            print(f" {effect_interp}")
            print(f"  Sample: {int(row['topic_n'])} reviews")
            print(f"  P-value: {row['p_value']:.2e}")
            print()
    else:
        print("  No topics with significant positive impact found.\n")

    print(f"\n{'NEGATIVE IMPACT TOPICS':^80}")
    print("=" * 80)
    print(f"Topics associated with significantly LOWER user ratings:\n")

    if len(negative_impact) > 0:
        for idx, row in negative_impact.iterrows():
            print(f"Topic {int(row['topic_id'])}: {row['keywords']}")
            print(f"  Average Rating: {row['topic_avg']:.2f}/5.0")
            print(f"  vs. Other Topics: {row['other_avg']:.2f}/5.0")
            print(f"  Difference: {row['topic_avg'] - row['other_avg']:.2f} stars (NEGATIVE)")
            print(f"  Effect Size (Cohen's d): {row['cohens_d']:.4f}", end="")

            # Interpret effect size
            abs_d = abs(row['cohens_d'])
            if abs_d < 0.2:
                effect_interp = "(negligible)"
            elif abs_d < 0.5:
                effect_interp = "(small)"
            elif abs_d < 0.8:
                effect_interp = "(medium)"
            else:
                effect_interp = "(large)"

            print(f" {effect_interp}")
            print(f"  Sample: {int(row['topic_n'])} reviews")
            print(f"  P-value: {row['p_value']:.2e}")
            print()
    else:
        print("  No topics with significant negative impact found.\n")

    # Not significant
    not_sig = test_results[~test_results['is_significant']]
    if len(not_sig) > 0:
        print(f"\nTopics with NO significant impact (p ≥ 0.05): {len(not_sig)}")

    return {
        'positive_impact': positive_impact,
        'negative_impact': negative_impact,
        'not_significant': not_sig
    }


# ============================================================================
# SECTION 7.7: VISUALIZATIONS
# ============================================================================

def visualize_topic_impact(test_results):
    """
    Create comprehensive impact visualizations
    """

    print("\nGenerating impact visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Average rating by topic
    ax = axes[0, 0]
    test_results_sorted = test_results.sort_values('topic_avg')
    colors = ['#e74c3c' if x < 3 else '#f39c12' if x < 4 else '#2ecc71'
              for x in test_results_sorted['topic_avg']]

    ax.barh(range(len(test_results_sorted)), test_results_sorted['topic_avg'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(test_results_sorted)))
    ax.set_yticklabels([f"Topic {int(t)}" for t in test_results_sorted['topic_id']])
    ax.set_xlabel('Average Rating')
    ax.set_title('Average Rating by Topic', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 5])
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, v in enumerate(test_results_sorted['topic_avg']):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center')

    # 2. P-values (significance)
    ax = axes[0, 1]
    test_results_sorted_p = test_results.sort_values('p_value')
    sig_colors = ['#2ecc71' if x < 0.05 else '#95a5a6'
                  for x in test_results_sorted_p['p_value']]

    ax.barh(range(len(test_results_sorted_p)), -np.log10(test_results_sorted_p['p_value']),
            color=sig_colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2,
               label='p = 0.05 threshold')
    ax.set_yticks(range(len(test_results_sorted_p)))
    ax.set_yticklabels([f"Topic {int(t)}" for t in test_results_sorted_p['topic_id']])
    ax.set_xlabel('-log10(p-value)')
    ax.set_title('Statistical Significance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # 3. Effect sizes (Cohen's d)
    ax = axes[1, 0]
    test_results_sorted_d = test_results.sort_values('cohens_d')
    d_colors = ['#e74c3c' if x < 0 else '#2ecc71'
                for x in test_results_sorted_d['cohens_d']]

    ax.barh(range(len(test_results_sorted_d)), test_results_sorted_d['cohens_d'],
            color=d_colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=-0.2, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
    ax.set_yticks(range(len(test_results_sorted_d)))
    ax.set_yticklabels([f"Topic {int(t)}" for t in test_results_sorted_d['topic_id']])
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title('Effect Size of Topic Impact', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 4. Sample sizes
    ax = axes[1, 1]
    x_pos = np.arange(len(test_results))
    width = 0.35

    ax.bar(x_pos - width / 2, test_results['topic_n'], width,
           label='With Keywords', alpha=0.7, color='#3498db')
    ax.bar(x_pos + width / 2, test_results['other_n'], width,
           label='Without Keywords', alpha=0.7, color='#95a5a6')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"T{int(t)}" for t in test_results['topic_id']])
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Sample Sizes by Topic', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('topic_impact_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: topic_impact_analysis.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_topic_impact_analysis(df, lda_model, tfidf_matrix, tfidf_vectorizer):
    """
    Execute complete topic impact analysis pipeline

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset (must have 'score' and 'content_clean' columns)
    lda_model : LatentDirichletAllocation
        Fitted LDA model
    tfidf_matrix : sparse matrix
        Document-term matrix
    tfidf_vectorizer : TfidfVectorizer
        Fitted vectorizer

    Returns:
    --------
    dict : All analysis results
    """

    print("\n" + "=" * 80)
    print("TOPIC IMPACT ANALYSIS - FEATURE IMPORTANCE & RATING CORRELATION")
    print("=" * 80)

    # Step 1: Assign dominant topics
    dominant_topics = assign_dominant_topics(df, lda_model, tfidf_matrix)

    # Step 2: Calculate average ratings by topic
    df_analysis, topic_stats = calculate_average_rating_by_topic(df, dominant_topics)

    # Step 3: Extract keywords
    topic_keywords = extract_topic_keywords(lda_model, tfidf_vectorizer, n_words=15)

    # Step 4: Calculate ratings by keyword presence
    df_analysis, topic_keywords, keyword_stats = calculate_rating_by_keyword_presence(
        df_analysis, topic_keywords, text_column='content_clean'
    )

    # Step 5: Perform statistical tests
    test_results = perform_statistical_tests(
        df_analysis, topic_keywords, text_column='content_clean'
    )

    # Step 6: Summarize findings
    impact_summary = summarize_impact_findings(test_results)

    # Step 7: Create visualizations
    visualize_topic_impact(test_results)

    # Step 8: Save results
    topic_stats.to_csv('topic_statistics.csv')
    keyword_stats.to_csv('keyword_presence_statistics.csv')
    test_results.to_csv('topic_impact_test_results.csv', index=False)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  1. topic_statistics.csv")
    print("  2. keyword_presence_statistics.csv")
    print("  3. topic_impact_test_results.csv")
    print("  4. topic_impact_analysis.png")

    return {
        'dominant_topics': dominant_topics,
        'topic_stats': topic_stats,
        'topic_keywords': topic_keywords,
        'keyword_stats': keyword_stats,
        'test_results': test_results,
        'impact_summary': impact_summary,
        'df_analysis': df_analysis
    }


# ============================================================================
# SECTION 8: KEY INSIGHTS EXTRACTION
# ============================================================================

def extract_key_insights(df):
    """
    Extract actionable business insights with expanded issue categories
    """
    print("\n" + "=" * 80)
    print("SECTION 8: KEY BUSINESS INSIGHTS")
    print("=" * 80)

    # 1. Most common issues in negative reviews (EXPANDED)
    print("\n1. TOP ISSUES IN NEGATIVE REVIEWS:")
    negative_reviews = df[df['sentiment_label'] == 'Negative']['content_clean_nostop']

    issue_keywords = {
        # Authentication & Access
        'Login/Authentication Issues': [
            'login', 'log in', 'sign in', 'signin', 'cant log', "can't log",
            'password', 'locked out', 'authentication', 'verify', 'verification',
            'forgot password', 'reset password', 'account access'
        ],

        # Performance & Stability
        'App Crashes/Freezing': [
            'crash', 'crashes', 'freeze', 'frozen', 'hang', 'stuck',
            'loading forever', 'stops working', 'force close', 'unresponsive',
            'black screen', 'white screen', 'not responding'
        ],

        'Slow Performance': [
            'slow', 'lag', 'laggy', 'sluggish', 'takes forever',
            'speed', 'performance', 'loading time', 'wait', 'waiting'
        ],

        # Billing & Payment
        'Billing/Charging Issues': [
            'charge', 'charged', 'bill', 'billing', 'payment', 'money',
            'fee', 'cost', 'expensive', 'overcharge', 'wrong charge',
            'unexpected charge', 'double charge', 'incorrect bill'
        ],

        'Auto Top-up Problems': [
            'auto topup', 'auto top up', 'automatic topup', 'auto payment',
            'keep charging', 'stop charging', 'deducted', 'taken money'
        ],

        # Data & Usage
        'Data Usage Issues': [
            'data', 'usage', 'data usage', 'using data', 'data gone',
            'data disappear', 'data vanish', 'eating data', 'data drain'
        ],

        'Balance/Credit Issues': [
            'balance', 'credit', 'top up', 'topup', 'recharge', 'reload',
            'cant see balance', 'balance missing', 'credit missing'
        ],

        # Functionality
        'Missing Features': [
            'missing', 'removed', 'gone', 'disappeared', 'unavailable',
            'cant find', "can't find", 'where is', 'no longer', 'taken away'
        ],

        'Feature Not Working': [
            'not working', 'doesnt work', "doesn't work", 'broken', 'error',
            'failed', 'fails', 'wont work', "won't work", 'cant use'
        ],

        # UI/UX
        'User Interface Problems': [
            'interface', 'ui', 'design', 'layout', 'confusing', 'complicated',
            'hard to use', 'difficult', 'cant navigate', 'poor design'
        ],

        # Updates & Versions
        'Update Problems': [
            'update', 'updated', 'new version', 'latest version', 'upgrade',
            'after update', 'since update', 'broken after', 'worse after'
        ],

        # Customer Service
        'Customer Service Issues': [
            'support', 'service', 'help', 'customer service', 'no help',
            'response', 'reply', 'contact', 'cant contact', 'no response',
            'unhelpful', 'rude', 'poor service'
        ],

        # Network & Connectivity
        'Network/Connection Issues': [
            'network', 'connection', 'signal', 'coverage', 'internet',
            'wifi', 'connect', 'disconnect', 'no service', 'poor signal'
        ],

        # Installation & Compatibility
        'Installation/Compatibility': [
            'install', 'download', 'wont install', 'cant install',
            'compatible', 'compatibility', 'phone', 'device', 'version'
        ],

        # Account Management
        'Account Issues': [
            'account', 'profile', 'settings', 'cant change', 'locked',
            'suspended', 'closed', 'cant access account'
        ]
    }

    issue_counts = {}
    issue_examples = {}

    for issue, keywords in issue_keywords.items():
        # Count mentions
        count = negative_reviews.apply(
            lambda x: any(keyword in str(x).lower() for keyword in keywords)
        ).sum()
        issue_counts[issue] = count

        # Get example review
        example_reviews = df[
            (df['sentiment_label'] == 'Negative') &
            (df['content_clean_nostop'].apply(
                lambda x: any(keyword in str(x).lower() for keyword in keywords)
            ))
            ]

        if len(example_reviews) > 0:
            issue_examples[issue] = example_reviews['content'].iloc[0][:150] + "..."

    # Create detailed dataframe
    issue_df = pd.DataFrame({
        'Issue Category': list(issue_counts.keys()),
        'Mentions': list(issue_counts.values()),
        'Percentage': [(count / len(negative_reviews) * 100) for count in issue_counts.values()],
        'Example': [issue_examples.get(issue, 'N/A') for issue in issue_counts.keys()]
    })

    issue_df = issue_df.sort_values('Mentions', ascending=False)

    print("\nDetailed Issue Breakdown:")
    print(issue_df[['Issue Category', 'Mentions', 'Percentage']].to_string(index=False))

    print("\n\nTop 5 Issues with Examples:")
    for idx, row in issue_df.head(5).iterrows():
        print(f"\n{row['Issue Category']}: {row['Mentions']} mentions ({row['Percentage']:.1f}%)")
        print(f"  Example: {row['Example']}")

    # 2. Sentiment by app version
    print("\n\n2. SENTIMENT BY APP VERSION:")
    version_sentiment = pd.crosstab(
        df['major_version'],
        df['sentiment_label'],
        normalize='index'
    ) * 100
    print(version_sentiment.round(2))

    # 3. Impact of company responses
    print("\n3. IMPACT OF COMPANY RESPONSES:")
    response_impact = df.groupby('has_reply')['score'].agg(['mean', 'count'])
    print(response_impact)

    # 4. Temporal trends
    print("\n4. SENTIMENT TRENDS OVER TIME:")
    monthly_sentiment = df.groupby([
        df['review_date'].apply(lambda x: x.strftime('%Y-%m')),
        'sentiment_label'
    ]).size().unstack(fill_value=0)
    print(monthly_sentiment.tail(6))

    return issue_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ONE NZ APP SENTIMENT ANALYSIS - BUSINESS ANALYTICS PROJECT")
    print("=" * 80)

    # Update this path with your file location
    FILE_PATH = 'my_one_nz_google_reviews.csv'

    # Execute analysis pipeline
    try:
        # Load data
        df = load_and_explore_data(FILE_PATH)

        # Preprocess
        df = preprocess_data(df)

        # EDA
        df = perform_eda(df)

        # Sentiment Analysis
        df = perform_sentiment_analysis(df)

        # Topic Modeling
        lda_model, nmf_model, tfidf_matrix, vectorizer = perform_topic_modeling(df)

        # Run topic impact analysis
        impact_results = run_topic_impact_analysis(
            df,
            lda_model,
            tfidf_matrix,
            vectorizer
        )

        # Access results
        print(impact_results['impact_summary']['negative_impact'])
        print(impact_results['test_results'])

        # ML Models
        ml_results, ml_vectorizer, y_test = build_ml_models(df)

        # Key Insights
        insights = extract_key_insights(df)

        # Save processed data
        df.to_csv('one_nz_processed_reviews.csv', index=False)
        print("\n\nProcessed data saved to 'one_nz_processed_reviews.csv'")

        # Save ML results for visualization
        print("\nSaving ML results for visualization...")
        import pickle

        with open('ml_results.pkl', 'wb') as f:
            pickle.dump(ml_results, f)
        print("ML results saved to 'ml_results.pkl'")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nNext step: Run the visualization notebook to generate all charts")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check your file path and data format.")