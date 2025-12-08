# models/geo_model.py
import joblib
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR) 


MODEL_PATH = os.path.join(ROOT_DIR, "local_kmeans_model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "local_scaler.pkl")
LABELS_PATH = os.path.join(ROOT_DIR, "cluster_labels.pkl")
GEOJSON_FILE = os.path.join(ROOT_DIR, "hotosm_idn_points_of_interest_points_geojson\hotosm_idn_points_of_interest_points_geojson.geojson")

class GeoPredictor:
    def __init__(self):
        print("--- Loading Geo Model Resources ---")
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.labels_map = joblib.load(LABELS_PATH)
            
            
            print("Loading GeoJSON data for feature extraction... (Wait a moment)")
            self.gdf = gpd.read_file(GEOJSON_FILE)
            self.gdf = self.gdf.to_crs(epsg=3857) # Ubah ke Meter
            print("Geo Model Ready!")
            
        except FileNotFoundError:
            print("ERROR: Model .pkl tidak ditemukan! Harap jalankan training dulu.")
            self.model = None

    def _get_features(self, lat, lon, radius=500):
        """Menghitung jumlah competitor & support di sekitar lat/lon"""
        # Buat Point dan ubah ke Meter (EPSG:3857)
        pt_geo = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        point_meter = pt_geo[0]
        
        # Buffer Circle
        buffer = point_meter.buffer(radius)
        neighbors = self.gdf[self.gdf.geometry.intersects(buffer)]
        
        if neighbors.empty:
            return 0, 0
            
        valid_neighbors = neighbors[neighbors['amenity'].notna()]
        
        
        COMPETITOR_TAGS = ["restaurant", "cafe", "fast_food"]
        SUPPORT_TAGS = ["bank", "school", "university", "office", "marketplace", "bus_station"]
        
        comp = valid_neighbors[valid_neighbors['amenity'].isin(COMPETITOR_TAGS)].shape[0]
        supp = valid_neighbors[valid_neighbors['amenity'].isin(SUPPORT_TAGS)].shape[0]
        
        return comp, supp

    def predict_score(self, lat, lon):
        """
        Input: Lat, Lon
        Output: Score (0.0 - 1.0) untuk keperluan TOPSIS
        """
        if self.model is None:
            return 0.1 # Default bad score jika model error
            
        # 1. Hitung fitur realtime
        comp, supp = self._get_features(lat, lon)
        
        # 2. Prediksi Cluster
        features = np.array([[comp, supp]])
        features_scaled = self.scaler.transform(features)
        cluster_id = self.model.predict(features_scaled)[0]
        
        # 3. Mapping Cluster ke Skor
        # Kita ambil label text dari labels_map, misal: "High Potential (Strategis)"
        label_text = self.labels_map.get(cluster_id, "").lower()
        
        # Logika Konversi Cluster ID ke Skor Angka (Untuk TOPSIS)
        # Anda bisa sesuaikan ini berdasarkan hasil print training Anda mana cluster yg bagus
        if "high" in label_text:
            base_score = 0.9  # Sangat Bagus
        elif "medium" in label_text:
            base_score = 0.6  # Lumayan
        else: # Low / Sepi
            base_score = 0.3  # Kurang
            
        # Sedikit variasi berdasarkan jumlah support (agar tidak flat)
        final_score = base_score + (supp * 0.001) 
        
        return round(min(final_score, 1.0), 3)

