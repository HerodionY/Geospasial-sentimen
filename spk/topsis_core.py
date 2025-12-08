# spk/topsis_core.py
import pandas as pd
import numpy as np

# Definisi Bobot (Wj) dan Jenis Kriteria (Cost/Benefit)
# Di dunia nyata, bobot ini diperoleh dari AHP atau pakar
WEIGHTS = np.array([0.35, 0.30, 0.15, 0.20]) # W1, W2, W3, W4 (Total = 1.0)
CRITERIA_TYPE = np.array(['benefit', 'benefit', 'cost', 'cost']) # C1, C2, C3, C4

def run_topsis(data_df):
    """
    Menerapkan metode TOPSIS pada data kriteria.
    Dataframe harus memiliki kolom: C1_Geo, C2_Sentiment, C3_CompetitorDist, C4_Sewa.
    """
    X = data_df[['C1_Geo', 'C2_Sentiment', 'C3_CompetitorDist', 'C4_Sewa']].values

    R = X / np.sqrt(np.sum(X**2, axis=0))

    Y = R * WEIGHTS

    # 4. Penentuan Solusi Ideal Positif (A+) dan Negatif (A-)
    A_plus = np.zeros(Y.shape[1])
    A_minus = np.zeros(Y.shape[1])

    for j in range(Y.shape[1]):
        if CRITERIA_TYPE[j] == 'benefit':
            A_plus[j] = np.max(Y[:, j])
            A_minus[j] = np.min(Y[:, j])
        else: # cost
            A_plus[j] = np.min(Y[:, j])
            A_minus[j] = np.max(Y[:, j])

    # 5. Perhitungan Jarak (Di+ dan Di-)
    D_plus = np.sqrt(np.sum((Y - A_plus)**2, axis=1))
    D_minus = np.sqrt(np.sum((Y - A_minus)**2, axis=1))

    # 6. Perhitungan Nilai Preferensi (Vi)
    data_df['Vi'] = D_minus / (D_minus + D_plus)
    
    # 7. Ranking
    data_df['Ranking'] = data_df['Vi'].rank(ascending=False).astype(int)
    
    return data_df.sort_values(by='Vi', ascending=False)