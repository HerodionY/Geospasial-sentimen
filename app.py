from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle 


from models.geo_model import run_geo_analysis
from models.sentiment_model import run_sentiment_analysis
from spk.topsis_core import run_topsis

app = Flask(__name__)

ALTERNATIVES = {
    'A1': {'lat': -6.2, 'lon': 106.8, 'sewa': 100},
    'A2': {'lat': -6.5, 'lon': 106.7, 'sewa': 80},
    'A3': {'lat': -6.1, 'lon': 106.9, 'sewa': 120}
}
# --- End Dummy Data ---

@app.route('/api/recommend', methods=['POST'])
def recommend_location():
    data = request.get_json()
    
    if not data or 'alternatives' not in data:
        return jsonify({"error": "Data input tidak valid. Butuh daftar alternatif."}), 400

    results = []
    
    try:
        # 1. LOOP ALTERNATIF UNTUK MENDAPATKAN C1 DAN C2 (AI Engine)
        for loc_id, loc_data in data['alternatives'].items():
            
            
            c1_score = run_geo_analysis(loc_data['lat'], loc_data['lon'])
            
            # 1b. Sentiment Engine (C2)
            # data_input: {text_review, area_id}
            c2_score = run_sentiment_analysis(loc_data['reviews'])
            
            # Kumpulkan semua kriteria (C1, C2, C3, C4)
            result = {
                'id': loc_id,
                'C1_Geo': c1_score,
                'C2_Sentiment': c2_score,
                'C3_CompetitorDist': loc_data['competitor_dist'],
                'C4_Sewa': loc_data['sewa_cost'] 
            }
            results.append(result)

        # 2. SPK Core (TOPSIS)
        # Hasil dari AI Engine (results) dimasukkan ke TOPSIS
        final_ranking_df = run_topsis(pd.DataFrame(results))
        
        # 3. Output
        return jsonify({
            "code": 200,
            "status": "success",
            "message": "Rekomendasi lokasi berhasil dihitung.",
            "data": final_ranking_df.to_dict(orient='records')
        })

    except Exception as e:
        app.logger.error(f"Error pada proses rekomendasi: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Untuk development, atur debug=True
    app.run(debug=True, port=5000)