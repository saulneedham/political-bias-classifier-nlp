# Final Training and Model Save
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib # Required for saving the model

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

# Define raw input and target variables (using all data)
X = df['Text']
y = df['Bias']

# --- 2. Define the Pipeline (Reusing Best Configuration) ---

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

# Create the final pipeline object
textClf = Pipeline([
    ('tfidf', vectorizerStep),
    ('clf', modelStep),
])

# --- 3. Final Training and Saving ---

print("--- Final Model Training (100% Data) ---")
# Train the entire Pipeline on ALL available data (X and y)
textClf.fit(X, y)

print("--- Saving Model Assets ---")
# Save the trained Pipeline object
joblib.dump(textClf, 'final_bias_pipeline.pkl')
print("Complete Pipeline saved as 'final_bias_pipeline.pkl'")
