import pandas as pd
import numpy as np

def run_saw(df):
    """
    Fungsi Utama Perhitungan SAW (Simple Additive Weighting).
    
    Input: DataFrame dengan kolom kriteria (C1_Geo, C2_Sentiment, dll)
    Output: DataFrame yang sudah diurutkan berdasarkan skor tertinggi.
    """
    
    # --- 1. KONFIGURASI BOBOT & KRITERIA ---
    # Pastikan nama key (C1_Geo, dll) SAMA PERSIS dengan yang dikirim dari app.py
    
    # Tentukan Bobot (Total harus 1.0 atau 100%)
    # Anda bisa ubah angka ini sesuai prioritas bisnis
    weights = {
        'C1_Geo': 0.35,            # 35% - Lokasi Strategis
        'C2_Sentiment': 0.25,      # 25% - Review Bagus
        'C3_CompetitorDist': 0.20, # 20% - Jarak Kompetitor
        'C4_Sewa': 0.20            # 20% - Harga Sewa
    }
    
    # Tentukan Jenis Kriteria ('benefit' atau 'cost')
    criteria_type = {
        'C1_Geo': 'benefit',            # Semakin besar skor, semakin bagus
        'C2_Sentiment': 'benefit',      # Semakin positif, semakin bagus
        'C3_CompetitorDist': 'cost',    # Semakin kecil (deka/sedikit), semakin bagus (atau sebaliknya tergantung strategi Anda)
        'C4_Sewa': 'cost'               # Semakin murah (kecil), semakin bagus
    }

    # Kita copy dataframe agar data asli tidak rusak
    df_norm = df.copy()
    
    # --- 2. PROSES NORMALISASI MATRIKS ---
    # Rumus SAW:
    # Benefit = Nilai / Max
    # Cost    = Min / Nilai
    
    for col, c_type in criteria_type.items():
        # Cek apakah kolom ini ada di data yang dikirim app.py?
        if col not in df.columns:
            print(f"Warning: Kolom {col} tidak ditemukan di data input.")
            continue
            
        # Ambil nilai Max dan Min dari kolom tersebut
        max_val = df[col].max()
        min_val = df[col].min()
        
        # Mencegah pembagian dengan nol (Error Division by Zero)
        if max_val == 0: max_val = 1
        if min_val == 0: min_val = 0.001

        if c_type == 'benefit':
            # Rumus Normalisasi Benefit
            df_norm[col] = df[col] / max_val
            
        elif c_type == 'cost':
            # Rumus Normalisasi Cost
            # Gunakan lambda function untuk menangani jika ada nilai 0 di baris data
            df_norm[col] = df[col].apply(lambda x: min_val / x if x > 0 else 0)

    # --- 3. PERHITUNGAN SKOR AKHIR (PERANGKINGAN) ---
    # Skor = (Bobot * NilaiNormalisasi) + ...
    
    df['SAW_Score'] = 0.0
    
    for col, weight in weights.items():
        if col in df_norm.columns:
            # Tambahkan nilai terbobot ke skor akhir
            df['SAW_Score'] += df_norm[col] * weight
            
    # --- 4. OUTPUT ---
    # Urutkan dari skor tertinggi ke terendah
    final_df = df.sort_values(by='SAW_Score', ascending=False).reset_index(drop=True)
    
    # Opsional: Tampilkan kolom 'SAW_Score' agar user tahu nilainya
    # Kita kembalikan DataFrame asli + kolom skor
    return final_df