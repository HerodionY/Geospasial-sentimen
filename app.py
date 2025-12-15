

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import traceback


from models.sawGeoModel import GeoTOPSISPredictor
from models.sentiment_model import run_sentiment_analysis

app = Flask(__name__)
CORS(app)  


FACILITY_MAP = {
    "school": "🎓 Pendidikan",
    "university": "🎓 Kampus",
    "office": "🏢 Area Kantor",
    "bank": "🏦 Perbankan",
    "atm": "🏧 Akses ATM",
    "mall": "🛍️ Dekat Mall",
    "supermarket": "🛒 Belanja Harian",
    "hospital": "🏥 RS/Klinik",
    "clinic": "🏥 RS/Klinik",
    "residential": "🏠 Pemukiman",
    "fuel": "⛽ SPBU",
    "parking": "🅿️ Area Parkir",
    "bus_station": "🚌 Terminal Bus",
    

    "station": "🚉 Stasiun Kereta",
    "train_station": "🚉 Stasiun Kereta",
    "airport": "✈️ Bandara",
    "aerodrome": "✈️ Bandara"
}


BUSINESS_CONFIG = {
    "restaurant": { 
        "competitor": ["restaurant", "cafe", "fast_food"], 
        "support": ["office", "school", "university", "bank", "station", "mall"] 
    },
    "minimarket": { 
        "competitor": ["convenience", "supermarket"], 
        "support": ["school", "residential", "bank", "station"] 
    },
    "workshop": { 
        "competitor": ["car_repair", "tyres"], 
        "support": ["fuel", "parking"] 
    },
    "pharmacy": { 
        "competitor": ["pharmacy"], 
        "support": ["hospital", "clinic"] 
    },
    "electronics": { 
        "competitor": ["electronics", "mobile_phone"], 
        "support": ["mall", "atm"] 
    }
}


print("Sedang memuat Geo Engine...")
geo_engine = GeoTOPSISPredictor()
print("Sistem Siap!")



def generate_tags(raw_data):
    """
    Membaca data lapangan dari GeoEngine dan membuat label pendek (Tags).
    """
    tags = []
    
 
    for key, label in FACILITY_MAP.items():
        if raw_data.get(key, 0) > 0:
            if label not in tags: # Mencegah duplikat
                tags.append(label)
    
    # 2. Tag tambahan berdasarkan keramaian global
    support_total = raw_data.get('support_count', 0)
    if support_total > 15:
        tags.insert(0, "🔥 Hotspot Ramai") # Taruh di depan
    elif support_total > 0 and len(tags) == 0:
        tags.append("✅ Ada Fasilitas Publik")

    return tags

def generate_explanation(geo_data, sewa_cost, sentiment_score):
    reasons = []
    
    # 1. Analisa Fasilitas
    supports = geo_data['raw_data'].get('support_count', 0)
    if supports > 20:
        reasons.append(f"🔥 Sangat Strategis: Dikelilingi {supports} fasilitas publik.")
    elif supports > 5:
        reasons.append(f"✅ Cukup Strategis: Ada {supports} fasilitas pendukung.")
    else:
        reasons.append(f"⚠️ Area Sepi: Hanya ditemukan {supports} fasilitas publik.")

    # 2. Analisa Kompetitor
    dist = geo_data['raw_data'].get('competitor_dist', 0)
    if dist > 2000:
        reasons.append("💎 Peluang Monopoli: Tidak ada pesaing dalam radius 2KM.")
    elif dist > 500:
        reasons.append("✅ Aman: Pesaing terdekat berjarak cukup jauh.")
    else:
        reasons.append(f"⚔️ Persaingan Ketat: Pesaing hanya berjarak {int(dist)} meter.")

    # 3. Analisa Harga Sewa
    if sewa_cost > 150000000:
        reasons.append("💰 Modal Besar: Biaya sewa tinggi.")
    elif sewa_cost < 50000000:
        reasons.append("🏷️ Hemat Biaya: Sewa terjangkau.")

    # 4. Analisa Sentimen
    if sentiment_score > 0.7:
        reasons.append("⭐ Reputasi Bagus: Review lokasi positif.")
    
    return reasons


@app.route('/api/recommend', methods=['POST'])
def recommend_location():
    data = request.get_json()
    
    # Validasi Input
    if not data or 'alternatives' not in data:
        return jsonify({"error": "Data input tidak valid. Field 'alternatives' wajib ada."}), 400

    biz_type = data.get('business_type', 'restaurant')
    if biz_type not in BUSINESS_CONFIG:
        return jsonify({"error": f"Tipe bisnis '{biz_type}' tidak dikenali."}), 400
        
    config = BUSINESS_CONFIG[biz_type]
    results = []
    
    try:
        # Loop setiap alternatif lokasi yang dikirim user
        for loc_id, loc_data in data['alternatives'].items():
            
            # A. PANGGIL GEO ENGINE
            # Engine akan mencari jumlah support & jarak kompetitor
            geo_result = geo_engine.predict_score(
                loc_data['lat'], 
                loc_data['lon'],
                config['competitor'], 
                config['support']     
            )
            
            # B. PANGGIL SENTIMENT ENGINE
            sent_score = run_sentiment_analysis(loc_data.get('reviews', []))
            
            # C. HITUNG HARGA (Normalisasi Cost)
            sewa = loc_data.get('sewa_cost', 100000000) 
           
            norm_sewa = 100000000 / (sewa if sewa > 0 else 1)
            if norm_sewa > 1: norm_sewa = 1 # Cap di 1
            if norm_sewa < 0: norm_sewa = 0
            
            # D. HITUNG FINAL SCORE (SAW / TOPSIS Sederhana)
            # Bobot: Geo (60%), Sentiment (30%), Harga (10%)
            geo_score = geo_result['score']
            final_vi = (geo_score * 0.6) + (sent_score * 0.3) + (norm_sewa * 0.1)
            
            # E. GENERATE TEXT EXPLANATION & TAGS
            explanation_list = generate_explanation(geo_result, sewa, sent_score)
            location_tags = generate_tags(geo_result['raw_data'])
            
            # F. SUSUN DATA PER LOKASI
            results.append({
                'id': loc_id,
                'Business_Type': biz_type,
                'Final_Vi': round(final_vi, 4),
                'Scores': {
                    'Geo': round(geo_score, 2),
                    'Sentiment': round(sent_score, 2),
                    'Price_Value': round(norm_sewa, 2)
                },
                'Data_Lapangan': geo_result['raw_data'],
                'Analisa_AI': explanation_list,         
                'Tags': location_tags                    
            })

        df = pd.DataFrame(results)
        if not df.empty:
            df['Ranking'] = df['Final_Vi'].rank(ascending=False, method='min').astype(int)
            df_sorted = df.sort_values(by='Ranking', ascending=True)
            result_list = df_sorted.to_dict(orient='records')
        else:
            result_list = []
        
        return jsonify({
            "code": 200,
            "status": "success",
            "message": "Rekomendasi berhasil dihitung.",
            "data": result_list
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)