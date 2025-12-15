# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np

# # Import Class yang baru kita buat
# from models.geo_model import GeoPredictor
# from models.sentiment_model import run_sentiment_analysis
# from spk.topsis_core import run_topsis

# app = Flask(__name__)


# print("Sedang memuat Geo AI Model...")
# geo_engine = GeoPredictor()
# print("Sistem Siap!")

# @app.route('/api/recommend', methods=['POST'])
# def recommend_location():
#     data = request.get_json()
    
#     if not data or 'alternatives' not in data:
#         return jsonify({"error": "Data input tidak valid. Butuh daftar alternatif."}), 400

#     results = []
    
#     try:

#         for loc_id, loc_data in data['alternatives'].items():
            
#             # --- BAGIAN GEOGRAPHIC AI ---
#             # Kita panggil method predict_score dari geo_engine yang sudah diload
#             c1_score = geo_engine.predict_score(loc_data['lat'], loc_data['lon'])
            
#             # --- BAGIAN SENTIMENT AI ---
#             # (Asumsi sentiment model Anda sudah jalan)
#             c2_score = run_sentiment_analysis(loc_data.get('reviews', []))
            
#             # Kumpulkan semua kriteria
#             result = {
#                 'id': loc_id,
#                 'C1_Geo': c1_score,         # Benefit (Semakin tinggi semakin bagus)
#                 'C2_Sentiment': c2_score,             # Benefit
#                 'C3_CompetitorDist': loc_data['competitor_dist'], # Cost (Biasanya jarak kompetitor: makin jauh makin bagus, atau density makin kecil makin bagus?) -> Cek logika TOPSIS Anda
#                 'C4_Sewa': loc_data['sewa_cost']      # Cost (Murah makin bagus)
#             }
#             results.append(result)

#         # 2. SPK Core (TOPSIS)
#         df_results = pd.DataFrame(results)
        
        
#         final_ranking_df = run_topsis(df_results)
        
#         # 3. Output
#         return jsonify({
#             "code": 200,
#             "status": "success",
#             "message": "Rekomendasi lokasi berhasil dihitung.",
#             "data": final_ranking_df.to_dict(orient='records')
#         })

#     except Exception as e:
#         app.logger.error(f"Error pada proses rekomendasi: {e}")
#         # Print error lengkap ke terminal untuk debugging
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)




from flask import Flask, request, jsonify
import pandas as pd
from models.sawGeoModel import GeoTOPSISPredictor
from models.sentiment_model import run_sentiment_analysis
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- KAMUS TAGS (SAMA SEPERTI SEBELUMNYA) ---
BUSINESS_CONFIG = {
    "restaurant": { "competitor": ["restaurant", "cafe", "fast_food"], "support": ["office", "school", "university", "bank"] },
    "minimarket": { "competitor": ["convenience", "supermarket"], "support": ["school", "residential", "bank"] },
    "workshop":   { "competitor": ["car_repair", "tyres"], "support": ["fuel", "parking"] },
    "pharmacy":   { "competitor": ["pharmacy"], "support": ["hospital", "clinic"] },
    "electronics":{ "competitor": ["electronics", "mobile_phone"], "support": ["mall", "atm"] }
}

print("Sedang memuat Geo Engine...")
geo_engine = GeoTOPSISPredictor()
print("Sistem Siap!")

# --- FUNGSI BARU: PEMBUAT ALASAN ---
def generate_explanation(geo_data, sewa_cost, sentiment_score):
    reasons = []
    
    # 1. Analisa Geografis (Keramaian)
    supports = geo_data['raw_data']['support_count']
    if supports > 20:
        reasons.append(f"🔥 Sangat Strategis: Dikelilingi {supports} fasilitas publik (potensi ramai).")
    elif supports > 5:
        reasons.append(f"✅ Cukup Strategis: Ada {supports} fasilitas pendukung di sekitar.")
    else:
        reasons.append(f"⚠️ Area Sepi: Hanya ditemukan {supports} fasilitas publik.")

    # 2. Analisa Kompetitor (Saingan)
    dist = geo_data['raw_data']['competitor_dist']
    if dist > 2000:
        reasons.append("💎 Peluang Monopoli: Tidak ada pesaing dalam radius 2KM.")
    elif dist > 500:
        reasons.append("✅ Aman: Pesaing terdekat berjarak cukup jauh.")
    else:
        reasons.append(f"⚔️ Persaingan Ketat: Ada pesaing hanya berjarak {int(dist)} meter.")

    # 3. Analisa Harga Sewa
    # Asumsi mahal > 100 juta (bisa disesuaikan)
    if sewa_cost > 150000000:
        reasons.append("💰 Modal Besar: Biaya sewa tergolong tinggi.")
    elif sewa_cost < 50000000:
        reasons.append("🏷️ Hemat Biaya: Harga sewa sangat terjangkau.")

    # 4. Analisa Sentimen (Jika ada review)
    if sentiment_score > 0.7:
        reasons.append("⭐ Reputasi Bagus: Review lokasi positif.")
    
    return reasons

@app.route('/api/recommend', methods=['POST'])
def recommend_location():
    data = request.get_json()
    if not data or 'alternatives' not in data:
        return jsonify({"error": "Data input tidak valid."}), 400

    biz_type = data.get('business_type', 'restaurant')
    if biz_type not in BUSINESS_CONFIG:
        return jsonify({"error": "Tipe bisnis tidak valid"}), 400
        
    config = BUSINESS_CONFIG[biz_type]

    results = []
    
    try:
        for loc_id, loc_data in data['alternatives'].items():
            
            # PANGGIL ENGINE (Sekarang return Dictionary)
            geo_result = geo_engine.predict_score(
                loc_data['lat'], 
                loc_data['lon'],
                config['competitor'], 
                config['support']     
            )
            
            
            geo_score = geo_result['score']
            
            sent_score = run_sentiment_analysis(loc_data.get('reviews', []))
            
            sewa = loc_data.get('sewa_cost', 100)
            norm_sewa = 100 / (sewa if sewa > 0 else 1)
            if norm_sewa > 1: norm_sewa = 1
            
            
            final_vi = (geo_score * 0.6) + (sent_score * 0.3) + (norm_sewa * 0.1)
            
            
            explanation_list = generate_explanation(geo_result, sewa, sent_score)
            
            results.append({
                'id': loc_id,
                'Business_Type': biz_type,
                'Final_Vi': round(final_vi, 4),
                'Scores': {
                    'Geo': geo_score,
                    'Sentiment': sent_score,
                    'Price_Value': round(norm_sewa, 2)
                },
                'Data_Lapangan': geo_result['raw_data'], # Info jumlah sekolah/jarak pesaing
                'Analisa_AI': explanation_list           # <--- INI HASILNYA
            })

        # Ranking
        df = pd.DataFrame(results)
        df['Ranking'] = df['Final_Vi'].rank(ascending=False, method='min').astype(int)
        df_sorted = df.sort_values(by='Ranking', ascending=True)
        
        return jsonify({
            "code": 200,
            "status": "success",
            "data": df_sorted.to_dict(orient='records')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)