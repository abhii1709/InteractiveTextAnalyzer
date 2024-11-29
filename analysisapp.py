import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import nltk
import re

# Download necessary NLTK data
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit Page Configuration
st.set_page_config(
    page_title="Interactive Text Analysis Platform",
    page_icon="📝",
    layout="wide",
)

# Caching Functions to Enhance Performance
@st.cache_data
def load_sample_data():
    """Loads a small sample dataset for text analysis."""
    data = {
        "Text": [
            "Streamlit is an amazing framework for Python developers!",
            "The weather today is gloomy, but I feel happy.",
            "This product is terrible. I want a refund!",
            "Natural Language Processing is an exciting field.",
            "Streamlit makes it easy to build interactive web applications.",
        ]
    }
    return pd.DataFrame(data)

@st.cache_data
def preprocess_text(text):
    """Basic text preprocessing."""
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text

@st.cache_data
def generate_wordcloud(text_data):
    """Generate a word cloud from text data."""
    text = " ".join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wordcloud

@st.cache_data
def extract_keywords(text_data, top_n=10):
    """Extract top N keywords using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_n)
    X = vectorizer.fit_transform(text_data)
    keywords = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    return pd.DataFrame({"Keyword": keywords, "Score": scores}).sort_values(by="Score", ascending=False)

@st.cache_data
def perform_topic_modeling(text_data, num_topics=2, num_words=5):
    """Perform topic modeling using LDA."""
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(text_data)
    lda = LDA(n_components=num_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topics.append([words[i] for i in topic.argsort()[:-num_words - 1:-1]])
    return topics

# Streamlit App Layout
st.title("📝 Interactive Text Analysis Platform")
st.write("Analyze and visualize textual data with sentiment analysis, word clouds, keyword extraction, and topic modeling.")

# Sidebar for File Upload or Sample Data
st.sidebar.header("Upload or Use Sample Data")
data_source = st.sidebar.radio("Choose Data Source:", ("Sample Data", "Upload CSV"))

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    else:
        st.sidebar.info("Please upload a CSV file.")
        st.stop()
else:
    df = load_sample_data()

# Display Dataset
st.subheader("Dataset")
st.dataframe(df.head())

# Text Preprocessing
st.subheader("Preprocessed Text")
df["Cleaned_Text"] = df["Text"].apply(preprocess_text)
st.write(df[["Text", "Cleaned_Text"]])

# Sentiment Analysis
st.subheader("Sentiment Analysis")
if st.button("Analyze Sentiment"):
    sentiment_scores = df["Text"].apply(lambda x: sia.polarity_scores(x))
    df["Negative"] = sentiment_scores.apply(lambda x: x["neg"])
    df["Neutral"] = sentiment_scores.apply(lambda x: x["neu"])
    df["Positive"] = sentiment_scores.apply(lambda x: x["pos"])
    df["Compound"] = sentiment_scores.apply(lambda x: x["compound"])
    st.write(df[["Text", "Negative", "Neutral", "Positive", "Compound"]])

# Word Cloud Generation
st.subheader("Word Cloud")
if st.button("Generate Word Cloud"):
    wordcloud = generate_wordcloud(df["Cleaned_Text"])
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()

# Keyword Extraction
st.subheader("Keyword Extraction")
top_n = st.slider("Number of Keywords:", 5, 20, 10)
if st.button("Extract Keywords"):
    keywords = extract_keywords(df["Cleaned_Text"], top_n=top_n)
    st.write(keywords)

# Topic Modeling
st.subheader("Topic Modeling")
num_topics = st.slider("Number of Topics:", 2, 10, 3)
if st.button("Perform Topic Modeling"):
    topics = perform_topic_modeling(df["Cleaned_Text"], num_topics=num_topics)
    for i, topic in enumerate(topics):
        st.write(f"Topic {i + 1}: {', '.join(topic)}")

st.write("#### 🚀 Built with [Streamlit](https://streamlit.io/)")
