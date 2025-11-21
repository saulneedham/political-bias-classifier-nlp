import joblib
import pandas as pd
from sklearn.pipeline import Pipeline # Required for loading the saved Pipeline object

# --- 1. Define Test Articles (Balanced Spread & Consistent Length) ---

testArticles = [
    "The undeniable, urgent need for massive government investment in renewable energy far outweighs the temporary costs. Climate change inaction is a moral failure that disproportionately impacts vulnerable communities, demanding immediate and radical federal policy intervention and global regulatory changes.",
    
    "Reports from humanitarian organizations highlight the immense suffering and displacement resulting from the escalating conflict. A lasting peace will require significant diplomatic pressure and a commitment to address the root causes of the occupation, which often receive unbalanced coverage in Western media.",
    
    "Analysts suggest that the outcome of next month's complex trade negotiations will depend heavily on market reactions, supply chain stability, and evolving political priorities on both sides of the aisle. The consensus is that results will be mixed, requiring further data review.",
    
    "Economists are sounding the alarm over the financial solvency of green initiatives, suggesting that the drive for 'net zero' mandates has created market distortions and unsustainable subsidies, ultimately hurting the taxpayer.",
    
    "Crippling taxation and suffocating bureaucratic overreach are stifling small business growth and undermining the foundational principles of free-market capitalism. It's time to champion energy independence and lift restrictions that handcuff domestic industry.",
    
    "In a time of profound political turmoil and cultural decay, many conservative voters view the return of Donald Trump as the last line of defense against globalist elites and the hopelessly biased mainstream media establishment.",
 
    "A growing body of evidence confirms that systemic inequalities in wealth distribution are exacerbated by corporate lobbying and loopholes that allow the ultra-rich to shield assets, necessitating drastic changes to the tax code and regulatory oversight.",

    "The proposed public health mandate seeks to protect vulnerable populations, a move supported by most medical experts, though concerns about personal autonomy continue to fuel a spirited but minority opposition.",

    "The Department of Transportation released its quarterly report detailing infrastructure improvements across three states, noting delays in the western sector due to unforeseen supply chain disruptions and weather events, according to official statements.",

    "New reports on unsustainable deficit spending highlight a profound failure by Congress to manage the national debt, leaving taxpayers burdened and raising serious concerns about long-term economic and national security viability.",
    
    "The recent Supreme Court ruling represents a clear victory for individual property rights and a necessary rebuke of progressive judicial activism that threatens constitutional freedoms and the established checks and balances of our republic."
]

# --- 2. Prediction Function Setup ---

def classifyBiasList(articles: list, pipelinePath: str):
    """
    Loads saved ML assets and predicts bias for a list of articles.
    """
    try:
        loadedPipeline = joblib.load(pipelinePath)
    except FileNotFoundError:
        return "Error: Pipeline file not found. Ensure 03_final_model_save.py was run successfully."
    
    # predict the bias categories
    predictions = loadedPipeline.predict(articles)

    # create a clean results table
    resultsDf = pd.DataFrame({
        'Article Snippet': [a[:70] + '...' for a in articles],
        'Predicted Bias': predictions
    })
    
    return resultsDf

# --- Run Prediction ---

pipelineFilePath = 'final_bias_pipeline.pkl'

print("\n--- Running Single Source Predictions ---")

predictionResults = classifyBiasList(
    testArticles,
    pipelineFilePath
)

print(predictionResults)
