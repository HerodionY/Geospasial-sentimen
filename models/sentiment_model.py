# models/sentiment_model.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np


try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Mengunduh vader_lexicon NLTK...")
    nltk.download('vader_lexicon')

ANALYZER = SentimentIntensityAnalyzer()

def run_sentiment_analysis(reviews_list):
    """
    Menghitung skor sentimen pasar (C2) dari daftar ulasan.
    Skor adalah Compound Score rata-rata.
    """
    compound_scores = []
    
    if not reviews_list:
        return 0.5 # Netral jika tidak ada ulasan
        
    for review in reviews_list:
        vs = ANALYZER.polarity_scores(review)
        compound_scores.append(vs['compound'])
        
    # Ambil rata-rata Compound Score sebagai Skor Sentimen
    avg_compound = np.mean(compound_scores)
    
    c2_score = (avg_compound + 1) / 2
    
    return round(c2_score, 3)
