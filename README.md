# NLP vs LLM: Sentiment Analysis on Movie Reviews
This project explores and compares traditional Natural Language Processing (NLP) methods and modern Large Language Models (LLMs) for sentiment analysis using a dataset of movie reviews.

# 📂 Project Structure
Data Loading: Reads a dataset (moviereviews[1].csv) containing movie reviews and their sentiment labels.

Preprocessing: Applies data cleaning and preparation using NLP techniques (e.g., tokenization, stopword removal).

NLP Models: Implements traditional ML models like Naive Bayes, SVM, or Logistic Regression.

LLMs: Uses transformers (like BERT) to classify sentiment using pretrained language models.

Comparison: Evaluates models based on accuracy, precision, recall, and F1-score.

# 🧰 Technologies Used
Python

Pandas, NumPy

Scikit-learn

NLTK / SpaCy (for NLP preprocessing)

Transformers (HuggingFace)

Matplotlib / Seaborn (for visualizations)

# 🧪 How to Run
Clone this repository:

bash
Copy
Edit
git clone https://github.com/yourusername/NLP_vs_LLM.git
cd NLP_vs_LLM
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook: Open NLP_VS_LLM.ipynb in Jupyter or VS Code.

# 📊 Results
Includes performance comparisons between traditional models and transformer-based models, with visual insights on:

Accuracy trends

Model confusion matrices

Misclassified reviews

# 📁 Dataset
Ensure the dataset file moviereviews[1].csv is present in the root directory.

# 🚀 Future Work
Fine-tune transformer models on domain-specific data

Extend to multilingual sentiment analysis

Deploy as a web app or API
