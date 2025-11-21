# 81.57% bias1 Logistic Regression
# 62.92% bias2 Logistic Regression
# 65.89% bias1+2 Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# --- 1. Data Loading and Preprocessing ---
print("--- 1. Data Loading and Preprocessing ---")

# Load datasets
df1 = pd.read_csv('bias1.csv')
df2 = pd.read_csv('bias2.csv')

# --- 2. Data Standardization and Cleaning ---

# Standardize df1 columns and clean
df1.rename(columns={'Text': 'Text', 'Bias': 'Bias'}, inplace=True)
df1.dropna(subset=['Text', 'Bias'], inplace=True)
df1['Bias'] = df1['Bias'].replace({'leaning-left': 'lean left', 'leaning-right': 'lean right'})

# Standardize df2 columns and clean
df2.rename(columns={'page_text': 'Text', 'bias': 'Bias'}, inplace=True)
df2.dropna(subset=['Text', 'Bias'], inplace=True)
df2['Bias'] = df2['Bias'].replace({'leaning-left': 'lean left', 'leaning-right': 'lean right'})

# Create combined dataset
df_combined = pd.concat([df1[['Text', 'Bias']], df2[['Text', 'Bias']]], ignore_index=True)

# List of datasets to test
datasets = [
    ('Dataset 1 Only', df1),
    ('Dataset 2 Only', df2),
    ('Combined Data', df_combined)
]

# --- 3. Feature and Model Configuration ---

# TfidfVectorizer with Unigrams & Bigrams
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))

# Logistic Regression model setup
model = LogisticRegression(C=1.0, solver='liblinear', random_state=99, max_iter=1000, class_weight='balanced')
print("\nModel: Logistic Regression with Balanced Class Weights")
print("Features: TF-IDF with Unigrams & Bigrams (max 10,000)")


# --- 4. Iterative Training and Evaluation ---

print("\n--- 2. Running Comparative Tests (70% Train / 30% Test) ---")

for name, current_df in datasets:
    print(f"\n======== Running Test: {name} ========")

    # Define features and target
    X = current_df['Text']
    y = current_df['Bias']

    # Stratified Split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99, stratify=y
    )
    print(f"Total Samples: {len(current_df)} | Training: {len(X_train)} | Testing: {len(X_test)}")

    # Feature Engineering (Fit only on training data)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model Training
    model.fit(X_train_vec, y_train)

    # Prediction and Evaluation
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)

    print(f"ACCURACY ({name}): {round(accuracy, 4)} (or {round(accuracy * 100, 2)} %)")

    # Print a full report
    print("\nClassification Report:\n", classification_report(y_test, predictions, zero_division=0))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title(f'Logistic Regression CM - {name}')
    plt.show()
