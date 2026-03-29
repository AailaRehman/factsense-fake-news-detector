# FactSense — AI Fake News Detector

> End-to-end NLP project detecting misinformation using fine-tuned DistilBERT.
> Built as part of an AI/ML portfolio for fully-funded Masters applications in Europe.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.8%25-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🔗 Live Demo
**[Try FactSense live on HuggingFace Spaces →](https://huggingface.co/spaces/YOUR_USERNAME/factsense-fake-news-detector)**

---

## Results

| Model | Features | Validation Accuracy |
|-------|----------|-------------------|
| Naive Bayes | TF-IDF bigrams | 96.53% |
| Logistic Regression | TF-IDF bigrams | 98.95% |
| DistilBERT fine-tuned | Transformer embeddings | 98.80% |

---

## Project Structure
```
factsense-fake-news-detector/
├── fake_news_detector.ipynb        # Full pipeline notebook
├── app.py                          # Gradio web application
├── requirements.txt                # Dependencies
├── class_distribution.png          # EDA — class balance
├── text_length_dist.png            # EDA — article length analysis
├── wordclouds.png                  # EDA — vocabulary comparison
├── confusion_matrix_lr.png         # Baseline model results
├── confusion_matrix_distilbert.png # Final model results
└── README.md
```

---

## Methodology

### Phase 1 — Data Collection & EDA
- Loaded ISOT Fake News Dataset (44,898 articles) and WELFake Dataset (72,134 articles)
- Combined 116k+ articles from 5 diverse news sources
- Analyzed class distribution, text length patterns, and vocabulary differences
- Key finding: fake articles average **423 words** vs 385 for real news

### Phase 2 — Preprocessing & Baseline Models
- Built full text preprocessing pipeline (cleaning, stopword removal)
- TF-IDF vectorization with unigrams and bigrams (50k features)
- Trained Logistic Regression (98.95%) and Naive Bayes (96.53%) as baselines

### Phase 3 — DistilBERT Fine-tuning
- Fine-tuned `distilbert-base-uncased` using HuggingFace Transformers
- Discovered and corrected **source-level data leakage** in ISOT-only model
- Retrained on multi-source WELFake dataset for proper generalization
- Final accuracy: **98.8%** with correct real-world predictions

### Phase 4 — Deployment
- Built production Gradio web app with confidence gauge visualization
- Deployed permanently to HuggingFace Spaces (free hosting)

---

## Key Findings & Critical Analysis

- **Data leakage discovered:** Initial ISOT-only model achieved 100% accuracy
  but failed on real-world examples — it learned Reuters' writing style,
  not actual misinformation patterns
- **Fix applied:** Retrained on WELFake (4 diverse sources) — fixed generalization
- **DistilBERT vs TF-IDF:** Transformer captures semantic meaning and context
  that TF-IDF fundamentally cannot represent
- **Known limitation:** Model may struggle with subtle misinformation that
  mimics credible writing style — an inherent challenge in this research domain

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, HuggingFace Transformers |
| Classical ML | scikit-learn |
| NLP Preprocessing | NLTK, TF-IDF |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Deployment | Gradio, HuggingFace Spaces |
| Development | Google Colab T4 GPU (free tier) |

---

## Datasets

| Dataset | Articles | Sources |
|---------|----------|---------|
| ISOT Fake News | 44,898 | University of Victoria |
| WELFake | 72,134 | Kaggle, McIntire, Reuters, BuzzFeed |
| **Total** | **116,032** | **5 diverse sources** |

---

## How to Run Locally
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/factsense-fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## Author

**Aaila Rehman**
