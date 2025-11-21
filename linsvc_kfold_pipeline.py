# 70.70%avg Kfold + LinearSVC + Trigrams + Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import time

# --- 1. Data Loading and Preprocessing ---

# Load datasets
df1 = pd.read_csv('bias1.csv')
df2 = pd.read_csv('bias2.csv')

# Standardize df1 and df2 columns, clean, and concatenate
df1.rename(columns={'Text': 'Text', 'Bias': 'Bias'}, inplace=True)
df1.dropna(subset=['Text', 'Bias'], inplace=True)
df1['Bias'] = df1['Bias'].replace({'leaning-left': 'lean left', 'leaning-right': 'lean right'})

df2.rename(columns={'page_text': 'Text', 'bias': 'Bias'}, inplace=True)
df2.dropna(subset=['Text', 'Bias'], inplace=True)
df2['Bias'] = df2['Bias'].replace({'leaning-left': 'lean left', 'leaning-right': 'lean right'})

dfCombined = pd.concat([df1[['Text', 'Bias']], df2[['Text', 'Bias']]], ignore_index=True)
df = dfCombined

# Define raw input and target variables
X = df['Text']
y = df['Bias']

# --- 2. Define the Pipeline ---

# Step 1: Feature Extraction (Trigrams)
vectorizerStep = TfidfVectorizer(
    stop_words='english', 
    max_features=20000, 
    ngram_range=(1, 3)
)

# Step 2: Classifier (LinearSVC)
modelStep = LinearSVC(
    C=0.5, 
    random_state=99, 
    max_iter=2000, 
    class_weight='balanced'
)

# Create the pipeline object
textClf = Pipeline([
    ('tfidf', vectorizerStep),
    ('clf', modelStep),
])

# --- 3. K-Fold Cross-Validation ---
print("\n--- Running 5-Fold Cross-Validation with Pipeline ---")
startTime = time.time()

# K-Fold Setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

# Calculate the cross-validated scores using the Pipeline object
cvScores = cross_val_score(
    estimator=textClf, 
    X=X, 
    y=y,
    cv=kf,
    scoring='accuracy',
    n_jobs=-1
)

# --- 4. Output Results ---
print(f"Time taken for 5-Fold CV: {time.time() - startTime:.2f} seconds.")
print(f"\nIndividual Fold Accuracies (5 Folds): {np.round(cvScores, 4)}")
print("--------------------------------------------------")
print(f"AVERAGE 5-FOLD CV ACCURACY: {np.mean(cvScores):.4f} (or {np.mean(cvScores) * 100:.2f} %)")
print(f"ACCURACY ST. DEV: {np.std(cvScores):.4f}")
