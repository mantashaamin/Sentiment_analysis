# Sentiment Analysis of Google Play App Reviews Using NLP and Machine Learning

## 📌 Project Overview

This project aims to analyse user reviews of productivity and personal organisation applications available on the Google Play Store and classify them into three sentiment categories: **Positive, Negative, and Neutral**.

The project uses Natural Language Processing (NLP) techniques and traditional Machine Learning algorithms to understand user opinions and evaluate the performance of different classification models.

The primary objective is to compare different text representation techniques and machine learning algorithms to identify the most suitable approach for sentiment classification.

## 🎯 Objectives

- Collect and analyse Google Play Store reviews from productivity applications.
- Explore and understand review distributions, ratings and application-wise statistics.
- Clean and preprocess textual review data.
- Convert text into numerical features using Bag-of-Words, TF-IDF and Word2Vec.
- Train multiple traditional machine learning classification models.
- Compare models using standard evaluation metrics.
- Test model performance across different applications.
- Develop an interactive Streamlit interface for sentiment prediction.

## 📂 Dataset

The dataset contains Google Play Store reviews collected from **15 productivity and personal organisation applications**.

Important dataset attributes:

| Attribute | Description |
|---|---|
| `content` | Actual text of the user review |
| `score` | Rating given by the user (1–5 stars) |
| `appId` | Unique identifier of the application |

### Sentiment Classification

Sentiment labels are generated using the review ratings.

| Star Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

**Note:** These labels are rating-derived rather than manually annotated. Therefore, the assigned sentiment may not always match the emotion expressed in the review text.

## ⚙️ Project Methodology

The project follows these stages:

1. **Data Exploration:** Understand the dataset, review distribution, rating patterns and application-wise statistics.
2. **Data Cleaning:** Remove unnecessary columns, handle invalid records, resolve conflicting labels and remove repeated review combinations.
3. **Text Preprocessing:** Clean textual reviews, tokenize text and preserve important sentiment-related words such as negations.
4. **Feature Extraction:** Convert text into numerical representations using BoW, TF-IDF, CBOW and Skip-gram.
5. **Model Training:** Train and compare traditional machine learning algorithms using the extracted features.
6. **Model Evaluation:** Evaluate models using accuracy, precision, recall, F1-score and confusion matrices.
7. **Cross-Application Testing:** Examine how well trained models perform on reviews from different applications.
8. **Streamlit Application:** Build an interactive interface to predict sentiment from user-provided reviews.

## 🧹 Data Cleaning Results

| Description | Records |
|---|---:|
| Original dataset | 16,092 |
| Conflicting records removed | 429 |
| Repeated combinations removed | 5,024 |
| Final cleaned dataset | **10,639** |

### Final Cleaned Sentiment Distribution

| Sentiment | Records |
|---|---:|
| Positive | 4,228 |
| Negative | 3,387 |
| Neutral | 3,024 |
| **Total** | **10,639** |

## 📝 Text Preprocessing

The cleaned dataset was further processed using NLP techniques.

The preprocessing pipeline includes:

- Lowercase conversion.
- Removal of unnecessary punctuation and special characters.
- Tokenization.
- Stopword removal.
- Preservation of important negation and sentiment-related words.
- Removal of reviews that become empty after preprocessing.

The final preprocessed dataset contains **10,575 reviews**.

The original review text is retained alongside the processed text for analysis and duplicate-group identification.

## 🔢 Feature Extraction

Four feature representations have been implemented.

### 1. Bag-of-Words (BoW)

- Implemented using Scikit-learn's `CountVectorizer`.
- Represents text through word occurrence counts.
- Uses unigrams and bigrams.
- Maximum vocabulary size: 5,000 features.
- Minimum document frequency: 2.

### 2. TF-IDF

- Implemented using Scikit-learn's `TfidfVectorizer`.
- Assigns weights based on term frequency and inverse document frequency.
- Uses unigrams and bigrams.
- Maximum vocabulary size: 5,000 features.
- Minimum document frequency: 2.

### 3. Word2Vec – CBOW

- Implemented using Gensim.
- Predicts a target word using its surrounding context words.
- Embedding dimension: 100.
- Window size: 5.
- Minimum word frequency: 2.
- Training epochs: 20.

### 4. Word2Vec – Skip-gram

- Implemented using Gensim.
- Predicts surrounding context words using a target word.
- Embedding dimension: 100.
- Window size: 5.
- Minimum word frequency: 2.
- Training epochs: 20.

Word2Vec generates word-level embeddings. To obtain a fixed-size representation for each review, the average of its known word vectors is calculated.

### Feature Extraction Results

| Feature Representation | Training Matrix | Testing Matrix |
|---|---|---|
| Bag-of-Words | 8461 × 5000 | 2114 × 5000 |
| TF-IDF | 8461 × 5000 | 2114 × 5000 |
| Word2Vec CBOW | 8461 × 100 | 2114 × 100 |
| Word2Vec Skip-gram | 8461 × 100 | 2114 × 100 |

### Training and Testing Split

The preprocessed dataset was divided using an 80:20 group-aware train-test split.

| Dataset | Records |
|---|---:|
| Training | 8,461 |
| Testing | 2,114 |
| Total | 10,575 |

`GroupShuffleSplit` was used to keep identical original review texts in the same partition.

**Overlapping original review groups: 0**

This reduces the risk of data leakage caused by identical review texts appearing in both training and testing datasets.

## 🤖 Machine Learning Algorithms

The following algorithms are planned for comparison:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Decision Tree
- Random Forest

The models will be evaluated using the available feature representations.

Multinomial Naive Bayes will be evaluated with BoW and TF-IDF. It will not be used with Word2Vec embeddings because those embeddings can contain negative values.

The remaining four classifiers will be evaluated with all four feature representations.

This gives us **18 model-feature combinations** for comparison.

## 📊 Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics will help compare classification performance and identify the most suitable model.

## 📁 Repository Structure

```text
Sentiment_analysis/
│
├── app/
│   └── streamlit_app.py
│
├── dataset/
│   ├── raw/
│   └── processed/
│       └── features/
│
├── models/
│
├── research/
│
├── results/
│   └── plots/
│
├── src/
│   ├── data_exploration.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── DOCUMENT.txt
├── requirements.txt
├── .gitignore
└── README.md
```

Generated datasets, feature matrices and trained model files are excluded from GitHub to avoid unnecessarily large repository uploads.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Data visualization |
| NLTK | Text preprocessing |
| Scikit-learn | Feature extraction and machine learning |
| Gensim | Word2Vec embeddings |
| SciPy | Sparse matrix handling |
| Joblib | Saving and loading Python objects |
| Streamlit | Interactive application |
| Git and GitHub | Version control |
| Visual Studio Code | Development environment |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mantashaamin/Sentiment_analysis.git
cd Sentiment_analysis
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 5. Add the dataset

Place the original dataset inside:

```text
dataset/raw/google_play_reviews.csv
```

The dataset is not included in this repository. Users must obtain the dataset separately.

### 6. Run data exploration

```bash
python3 src/data_exploration.py
```

### 7. Run data cleaning

```bash
python3 src/data_cleaning.py
```

The cleaned dataset will be generated inside:

```text
dataset/processed/cleaned_reviews.csv
```

### 8. Run text preprocessing

```bash
python3 src/preprocessing.py
```

The preprocessed dataset will be generated inside:

```text
dataset/processed/preprocessed_reviews.csv
```

### 9. Run feature extraction

```bash
python3 src/feature_extraction.py
```

This generates the BoW, TF-IDF, CBOW and Skip-gram feature matrices and saves the corresponding vectorizers and Word2Vec models locally.

**Note:** Feature extraction must be completed before running the model-training pipeline.

## 📌 Current Project Progress

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Data Exploration | Completed |
| Phase 2 | Data Cleaning | Completed |
| Phase 3 | Text Preprocessing | Completed |
| Phase 4 | Feature Extraction | Completed |
| Phase 5 | Model Training | Upcoming |
| Phase 6 | Model Evaluation | Upcoming |
| Phase 7 | Cross-Application Testing | Upcoming |
| Phase 8 | Streamlit Application | Upcoming |

## ⚠️ Limitations

- Sentiment labels are derived from star ratings and may not always represent the actual emotion expressed in a review.
- The dataset focuses only on 15 productivity and personal organisation applications.
- Short or ambiguous reviews may be difficult to classify accurately.
- BoW and TF-IDF do not capture complete semantic context.
- Averaged Word2Vec embeddings lose word order and some contextual information.
- Model performance may vary when applied to applications outside the selected dataset.

## 🔮 Future Scope

- Experiment with advanced NLP techniques and contextual word embeddings.
- Explore deep learning and transformer-based sentiment classification.
- Expand the dataset to include applications from different categories.
- Improve classification of sarcasm, mixed opinions and context-dependent reviews.
- Deploy the application for real-world sentiment analysis.

## 👩‍💻 Contributors

Developed as an academic Machine Learning project.

**GitHub:** [mantashaamin](https://github.com/mantashaamin)

---

*This project is developed for educational and research purposes.*
