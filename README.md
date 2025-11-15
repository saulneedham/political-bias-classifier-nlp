# Political Bias Classifier (NLP)
A multi-class text classification model built with scikit-learn to systematically determine the political leaning of news articles. The final model utilizes LinearSVC and Trigram features to categorize text into five distinct bias levels. Validated with 5-Fold Cross-Validation, it achieves a reliable average accuracy of over 70%.

📘 Included Projects

Model Training & Validation – Code for data loading, feature engineering, model training, and performance validation using K-Fold Cross-Validation.

Final Model Assets – Saved production-ready files (`.pkl`) for the LinearSVC model and the fitted TfidfVectorizer.

Single-Source Predictor – A Python function to load the saved assets and classify new, raw article text into one of five categories.

Performance Metrics – Detailed classification reports and confusion matrices for model assessment.

🛠 Built With

Python

Scikit-learn

NumPy

Pandas

Joblib
