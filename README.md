# Truth Lens: Fake News Detection System

An automated machine learning and Natural Language Processing (NLP) tool designed to detect misinformation and classify news articles as true or fake.

## Features

- **Text Analysis**: Utilizes NLP techniques (SpaCy, NLTK) to process and analyze news text.
- **Machine Learning Models**: Implements predictive modeling (Logistic Regression, Transformers like BERT, TensorFlow/Keras) to determine text veracity.
- **Web Scraping**: Ability to scrape news articles from web sources for analysis using BeautifulSoup.
- **Image Processing**: Basic image processing capabilities with OpenCV and TensorFlow.

## Prerequisites

Ensure you have Python 3.8+ installed. All dependencies can be installed via `requirements.txt`.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbdlKabeer/truth-lens.git
   cd truth-lens
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK and SpaCy Data (if required by your script):**
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

## Usage

You can run the different analysis modules depending on your needs. For instance:

- **Fake News Modeling (Standard ML):**
  ```bash
  python fake_new.py
  ```

- **Fake News Modeling (SpaCy enhanced):**
  ```bash
  python fake_new_spacy.py
  ```

- **Transformer-based Text Analysis:**
  ```bash
  python text_analyse.py
  ```

- **Web Scraping:**
  ```bash
  python scrape.py
  ```

## Important Note

The raw `.csv` datasets (e.g., `news_dataset.csv`, `True.csv`, `Fake.csv`) and the serialized model (`fake_news_text_model.pkl`) are ignored in Git to save space and prevent large file push errors. Ensure you place your datasets in the project root before running the training scripts.

## License

MIT License
