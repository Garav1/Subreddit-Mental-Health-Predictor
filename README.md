````markdown
# 🧠 Subreddit Mental Health Predictor

A deep learning NLP-based project that uses Reddit posts to detect signs of mental health issues. This project utilizes transformer-based models like BERT to classify social media text into relevant mental health-related subreddits, acting as an early indicator of psychological distress.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## 🚀 Project Overview

The **Subreddit Mental Health Predictor** aims to use machine learning to classify Reddit posts based on their content, identifying which mental health community the post most likely belongs to. By analyzing posts from subreddits like `r/depression`, `r/anxiety`, and `r/mentalhealth`, we can gain insights into the mental state of users and potentially flag early warning signs.

---

## 📊 Dataset

The dataset includes Reddit posts with the following fields:

- `title`: Title of the Reddit post
- `selftext`: Main content of the post
- `created_utc`: Time of creation (in UTC)
- `over_18`: Boolean flag for NSFW content
- `subreddit`: Label indicating the target subreddit (used as the class)

We focus on the `selftext` column as the main feature and `subreddit` as the target label.

---

## 🧠 Model Architecture

- **Tokenizer:** BERT Tokenizer (`bert-base-uncased`)
- **Model:** Fine-tuned BERT (Bidirectional Encoder Representations from Transformers)
- **Classification Head:** Fully connected dense layer on top of BERT for multi-class classification
- **Metrics:** Accuracy, Precision, Recall, F1-Score

The model classifies text into subreddits associated with different mental health issues, effectively acting as a predictor of user intent or emotional state.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Garav1/Subreddit-Mental-Health-Predictor.git
cd Subreddit-Mental-Health-Predictor
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

To train and evaluate the model:

```bash
python train.py
```

To make predictions on new text:

```python
from predictor import predict_subreddit

text = "I've been feeling really anxious and can't sleep."
prediction = predict_subreddit(text)
print("Predicted Subreddit:", prediction)
```

> Ensure the model is trained before running predictions.

---

## 📈 Results

| Subreddit      | Precision | Recall | F1-Score |
| -------------- | --------- | ------ | -------- |
| r/depression   | 0.87      | 0.85   | 0.86     |
| r/anxiety      | 0.84      | 0.83   | 0.83     |
| r/mentalhealth | 0.82      | 0.84   | 0.83     |

*Model performance will vary based on dataset size, preprocessing, and model fine-tuning.*

---

## 📁 File Structure

```
Subreddit-Mental-Health-Predictor/
│
├── data/                      # Input dataset files
├── models/                    # Saved BERT models
├── notebooks/                 # Jupyter notebooks for EDA and experiments
├── predictor.py               # Script to load model and make predictions
├── train.py                   # Model training pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```

---

## ⚠️ Limitations

* Reddit posts may contain sarcasm, slang, or ambiguous content which can affect model accuracy.
* Classification is limited to a predefined set of subreddits.
* Model may not generalize well to other social media platforms or broader psychological analysis.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [PyTorch](https://pytorch.org/)
* [Reddit API / Pushshift](https://github.com/pushshift/api)

---

```
