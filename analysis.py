import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

# 1. LOAD DATA
# Ensure your CSV file is in the same folder as this script
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024 (1).csv')

# 2. PREPROCESSING
# Combining text columns into one 'corpus' for analysis
text_cols = ['Attack Type', 'Target Industry', 'Attack Source', 'Security Vulnerability Type']
df['corpus'] = df[text_cols].astype(str).agg(' '.join, axis=1)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Remove symbols/numbers
    return text

df['clean_corpus'] = df['corpus'].apply(clean_text)

# 3. VECTORIZATION (TF-IDF)
tfidf = TfidfVectorizer(max_features=100)
tfidf_matrix = tfidf.fit_transform(df['clean_corpus'])

# 4. TOPIC EXTRACTION (LDA)
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

# 5. CLUSTERING (K-MEANS)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# --- OUTPUT RESULTS ---
print("--- TOP WORDS PER LDA TOPIC ---")
feature_names = tfidf.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
    print(f"Topic {idx+1}: {', '.join(top_words)}")

print("\n--- K-MEANS CLUSTER SIZES ---")
print(df['cluster'].value_counts())

# Save the final results to a new CSV
df.to_csv('Analysis_Results.csv', index=False)
print("\nSuccess! Results saved to Analysis_Results.csv")
