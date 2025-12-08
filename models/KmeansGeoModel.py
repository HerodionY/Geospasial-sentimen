import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI ---
GEOJSON_FILE = "E:\code\TopsisSPPK\hotosm_idn_points_of_interest_points_geojson\hotosm_idn_points_of_interest_points_geojson.geojson"
MODEL_PATH = "local_kmeans_model.pkl"
SCALER_PATH = "local_scaler.pkl"

# Tags yang ingin kita analisis
# TARGET: Lokasi yang ingin kita nilai (misal: kita ingin melatih model berdasarkan lokasi Restoran yang sudah ada)
TARGET_AMENITY = "restaurant" 

# FITUR: Apa yang kita hitung di sekitarnya (Competitor & Support)
COMPETITOR_TAGS = ["restaurant", "cafe", "fast_food"]
SUPPORT_TAGS = ["bank", "school", "university", "office", "marketplace", "bus_station"]

class LocalGeoModel:
    def __init__(self, geojson_path):
        print(f"Loading dataset: {geojson_path} ...")
        self.gdf = gpd.read_file(geojson_path)
        
        # PENTING: GeoJSON biasanya dalam Lat/Lon (EPSG:4326).
        # Kita harus ubah ke Meter (Projected CRS) agar bisa buat radius 500m yang akurat.
        # Kita pakai EPSG:3395 (World Mercator) atau EPSG:3857 (Web Mercator)
        self.gdf = self.gdf.to_crs(epsg=3857)
        
        print(f"Total POI dalam dataset: {len(self.gdf)}")
        
        self.model = None
        self.scaler = None

    def calculate_surrounding_features(self, center_point, radius=500):
        """
        Menghitung jumlah fasilitas di sekitar titik (buffer) menggunakan GeoPandas.
        Jauh lebih cepat daripada API.
        """
        # 1. Buat lingkaran (buffer) di sekitar titik
        circle_buffer = center_point.buffer(radius)
        
        # 2. Cari semua titik yang masuk dalam lingkaran ini (Spatial Intersection)
        # Menggunakan spatial index (sindex) otomatis dari geopandas jika tersedia
        neighbors = self.gdf[self.gdf.geometry.intersects(circle_buffer)]
        
        if neighbors.empty:
            return 0, 0
            
        # 3. Hitung Kompetitor & Support berdasarkan kolom 'amenity'
        # (Pastikan kolom 'amenity' tidak kosong/NaN)
        neighbors_amenity = neighbors[neighbors['amenity'].notna()]
        
        competitor_count = neighbors_amenity[neighbors_amenity['amenity'].isin(COMPETITOR_TAGS)].shape[0]
        support_count = neighbors_amenity[neighbors_amenity['amenity'].isin(SUPPORT_TAGS)].shape[0]
        
        # Kurangi 1 competitor count (karena titik pusat itu sendiri ikut terhitung jika dia ada di dataset)
        competitor_count = max(0, competitor_count - 1)
        
        return competitor_count, support_count

    def train(self):
        print("\n--- Memulai Ekstraksi Fitur & Training ---")
        
        # 1. Filter data: Kita hanya melatih menggunakan lokasi yang memang 'TARGET_AMENITY'
        # Tujuannya: Belajar dari lokasi yang sudah terbukti ada bisnisnya.
        training_locations = self.gdf[self.gdf['amenity'] == TARGET_AMENITY]
        
        if training_locations.empty:
            print(f"Tidak ditemukan data dengan amenity='{TARGET_AMENITY}'. Coba ganti target.")
            return

        print(f"Menggunakan {len(training_locations)} lokasi '{TARGET_AMENITY}' sebagai data training.")
        
        features_data = []
        
        # Loop (Mungkin agak lama jika datanya ribuan, bisa dibatasi head(500) untuk tes)
        # Kita ambil sampel max 1000 agar tidak terlalu lama prosesnya
        sample_limit = min(1000, len(training_locations))
        training_subset = training_locations.sample(n=sample_limit, random_state=42)
        
        for idx, row in training_subset.iterrows():
            if idx % 100 == 0:
                print(f"Processing {idx}...")
                
            comp, supp = self.calculate_surrounding_features(row.geometry, radius=500)
            
            # Kita tidak pakai jarak ke pusat kota dulu agar universal di seluruh Indonesia
            features_data.append([comp, supp])
            
        # 2. Konversi ke Numpy
        X = np.array(features_data)
        
        # 3. Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 4. Training KMeans (Misal 3 Cluster: Sepi, Sedang, Ramai/Strategis)
        self.model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.model.fit(X_scaled)
        
        # 5. Analisis Hasil Cluster
        print("\n--- Hasil Cluster Centers (Scaled) ---")
        print("Urutan: [Competitor Count, Support Count]")
        centers_orig = self.scaler.inverse_transform(self.model.cluster_centers_)
        
        # Mari kita urutkan cluster dari yang "Paling Ramai" ke "Paling Sepi"
        # Kita asumsikan 'Support Count' yang tinggi = Lebih Strategis
        labels_map = {}
        sorted_indices = np.argsort(centers_orig[:, 1]) # Sort by support count
        
        print("\nInterpretasi Cluster:")
        cluster_names = ["Low Potential (Sepi)", "Medium Potential", "High Potential (Strategis)"]
        
        for i, original_idx in enumerate(sorted_indices):
            c_comp = centers_orig[original_idx][0]
            c_supp = centers_orig[original_idx][1]
            label_name = cluster_names[i]
            labels_map[original_idx] = label_name
            
            print(f"Cluster {original_idx}: Rata-rata {c_comp:.1f} Kompetitor, {c_supp:.1f} Fasilitas Pendukung -> {label_name}")
            
        # Simpan Model
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(labels_map, "cluster_labels.pkl")
        print(f"\nModel disimpan ke {MODEL_PATH}")

    def predict_coordinate(self, lat, lon):
        """
        Fungsi untuk memprediksi lokasi baru (bukan dari dataset)
        """
        # Load resources jika belum ada
        if self.model is None:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.labels_map = joblib.load("cluster_labels.pkl")
            
        # Buat Point geometri (Ingat urutan: Lon, Lat)
        from shapely.geometry import Point
        
        # Kita harus transform koordinat input (Lat/Lon) ke Meter (EPSG:3857)
        # Cara manual sederhana atau pakai GeoSeries transform
        pt_geo = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857)
        point_meter = pt_geo[0]
        
        # Hitung fitur
        comp, supp = self.calculate_surrounding_features(point_meter, radius=500)
        
        # Prediksi
        features = np.array([[comp, supp]])
        features_scaled = self.scaler.transform(features)
        cluster_id = self.model.predict(features_scaled)[0]
        
        return {
            "lokasi": f"{lat}, {lon}",
            "fitur": {"competitors_500m": comp, "support_500m": supp},
            "prediksi_cluster": cluster_id,
            "kesimpulan": self.labels_map.get(cluster_id, "Unknown")
        }

# --- JALANKAN ---
if __name__ == "__main__":
    # Inisialisasi
    geo_model = LocalGeoModel(GEOJSON_FILE)
    
    # 1. Training (Jalankan sekali saja)
    geo_model.train()
    
    # 2. Test Prediksi Lokasi Baru (Misal koordinat sembarang di area dataset)
    # Contoh Koordinat (pastikan koordinat ini berada di area coverage GeoJSON Anda)
    # Ganti dengan Lat/Lon yang relevan dengan isi GeoJSON
    test_lat = -6.971841  # Diambil dari snippet file Anda (daerah Kuningan/Cirebon?)
    test_lon = 108.519947 
    
    print("\n--- Test Prediksi Lokasi Baru ---")
    hasil = geo_model.predict_coordinate(test_lat, test_lon)
    print(hasil)