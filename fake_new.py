import re
import cv2
import nltk
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

Download stopwords for NLP
nltk.download('stopwords')
nltk.download('punkt')



class FakeNewsDetector:
    def __init__(self):
        self.text_model = self.load_text_model()
        self.image_model = self.load_image_model()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    def load_text_model(self):
        """Load or train a simple text-based fake news classifier."""
        try:
            return load_model("fake_news_text_model.h5")
        except:
            return self.train_text_model()

    def train_text_model(self):
        """Train a basic text classification model."""
        df = pd.read_csv("news_dataset.csv")  # Dataset with 'text' and 'label' (fake/real)
        X, y = df['text'], df['label']

        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_transformed = vectorizer.fit_transform(X)

        model = make_pipeline(TfidfVectorizer(stop_words='english', max_features=5000), LogisticRegression())
        model.fit(X, y)

        return model

    def preprocess_text(self, text):
        """Basic text cleaning and tokenization."""
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    def analyze_text(self, text):
        """Predict if the text is fake news."""
        processed_text = self.preprocess_text(text)
        return self.text_model.predict([processed_text])[0]

    def load_image_model(self):
        """Load a pre-trained deep learning model for image verification."""
        try:
            return tf.keras.models.load_model("fake_news_image_model.h5")
        except:
            return None

    def analyze_image(self, image_path):
        """Predict if an image is fake/manipulated using deep learning."""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        if self.image_model:
            prediction = self.image_model.predict(img)
            return "Fake Image" if prediction > 0.5 else "Real Image"
        return "No Image Model Available"

    def check_metadata(self, url):
        """Scrape website metadata to assess source credibility."""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else "No Title"
            return f"Website Title: {title}"
        except:
            return "Unable to fetch metadata"

    def detect_fake_news(self, text, image_path=None, url=None):
        """Run multi-modal detection for fake news."""
        text_result = self.analyze_text(text)
        image_result = self.analyze_image(image_path) if image_path else "No Image Provided"
        metadata_result = self.check_metadata(url) if url else "No URL Provided"

        return {
            "Text Analysis": "Fake News" if text_result else "Real News",
            "Image Analysis": image_result,
            "Metadata Check": metadata_result
        }

# Example Usage
detector = FakeNewsDetector()
result = detector.detect_fake_news(
    text="Breaking: Nigerian government bans all social media platforms...",
    image_path=None,
    url="https://guardian.ng"
)
print(result)



#!!================= KAGGLE FAKE DATASET LOADING START ============================
# import pandas as pd

# # Load the dataset
# fake_df = pd.read_csv("Fake.csv")
# real_df = pd.read_csv("True.csv")

# # Add labels (0 for Real, 1 for Fake)
# fake_df["label"] = 1
# real_df["label"] = 0

# # Combine both datasets
# df = pd.concat([fake_df, real_df], ignore_index=True)

# # Shuffle the dataset
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Save the dataset
# df.to_csv("news_dataset.csv", index=False)

#!!================= KAGGLE FAKE DATASET LOADING START ============================