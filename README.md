
# Sentiment Analysis of Google Play App Reviews Using NLP and Machine Learning

## 📌 Project Overview

This project aims to analyse user reviews of productivity and personal organisation applications available on the Google Play Store and classify them into three sentiment categories: **Positive, Negative, and Neutral**.

The project uses Natural Language Processing (NLP) techniques and traditional Machine Learning algorithms to understand user opinions and evaluate the performance of different classification models.

The primary objective is to compare multiple machine learning algorithms and identify which performs best for sentiment classification.

## 🎯 Objectives

- Collect and analyse Google Play Store reviews from productivity applications.
- Clean and preprocess textual review data.
- Convert text into numerical features using Bag-of-Words and TF-IDF.
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

**Note:** These labels are rating-derived rather than manually annotated.

## ⚙️ Project Methodology

The project follows these stages:

1. **Data Exploration:** Understand the dataset, review distribution, rating patterns, and application-wise statistics.
2. **Data Cleaning:** Remove unnecessary columns, handle invalid records, resolve conflicting labels, and remove repeated review combinations.
3. **Text Preprocessing:** Prepare textual reviews for feature extraction using NLP techniques.
4. **Feature Extraction:** Convert text into numerical representations using Bag-of-Words and TF-IDF.
5. **Model Training:** Train and compare traditional machine learning algorithms.
6. **Model Evaluation:** Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
7. **Cross-Application Testing:** Examine how well trained models perform on reviews from different applications.
8. **Streamlit Application:** Build an interactive interface to predict sentiment from user-provided reviews.

## 🤖 Machine Learning Algorithms

The following algorithms are planned for comparison:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)
- Decision Tree
- Random Forest

## 📊 Current Dataset Cleaning Results

| Description | Records |
|---|---:|
| Original dataset | 16,092 |
| Conflicting records removed | 429 |
| Repeated combinations removed | 5,024 |
| Final cleaned dataset | **10,639** |

### Final Sentiment Distribution

| Sentiment | Records |
|---|---:|
| Positive | 4,228 |
| Negative | 3,387 |
| Neutral | 3,024 |
| **Total** | **10,639** |

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
└── README.md
```

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Natural Language Processing:** NLTK
- **Machine Learning:** Scikit-learn
- **Application Development:** Streamlit
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub

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
pip install -r requirements.txt
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

## 📈 Evaluation Metrics

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics will help compare model performance and identify the most suitable classifier.

## ⚠️ Limitations

- Sentiment labels are derived from star ratings and may not always represent the actual emotion expressed in the review.
- The dataset focuses only on 15 productivity and personal organisation applications.
- Short or ambiguous reviews may be difficult to classify accurately.
- Model performance may vary when applied to applications outside the selected dataset.

## 🔮 Future Scope

- Experiment with advanced NLP techniques and word embeddings.
- Explore deep learning and transformer-based sentiment classification.
- Expand the dataset to include applications from different categories.
- Improve classification of sarcasm, mixed opinions, and context-dependent reviews.
- Deploy the application for real-world sentiment analysis.

## 👩‍💻 Contributors

Developed as an academic Machine Learning project.

**GitHub:** [mantashaamin](https://github.com/mantashaamin)

---

*This project is developed for educational and research purposes.*
