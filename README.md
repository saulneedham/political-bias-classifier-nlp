# Political Bias Classifier (NLP)
A multi-class text classification model built with scikit-learn to systematically determine the political leaning of news articles. The final model utilizes LinearSVC and Trigram features to categorize text into five distinct bias levels. Validated with 5-Fold Cross-Validation, it achieves a reliable average accuracy of over 70%.

📘 Included Projects

01_logistic_regression_comparison.py – Benchmarks the initial models performance across individual and combined datasets for comparison

02_linsvc_kfold_pipeline.py – Validates the final chosen model (LinearSVC + Trigrams) and methodology using 5-Fold Cross-Validation

03_final_model_save.py – Trains the final classification pipeline on 100% of the data and saves the complete model asset for deployment

04_predict_new_articles.py – Loads the saved model and demonstrates its real-world bias classification capability on new, unseen text samples

📚 Data Sources

Bias Dataset 1 (`bias1.csv`):
    Title: Political Bias
    Author: Mayobanex Santana
    Link: https://www.kaggle.com/datasets/mayobanexsantana/political-bias

Bias Dataset 2 (`bias2.csv`):
    Title: News articles for political bias classification
    Author: Pablo Gandia
    Link: https://www.kaggle.com/datasets/gandpablo/news-articles-for-political-bias-classification

🛠 Built With

Python

Scikit-learn

NumPy

Pandas

Joblib
