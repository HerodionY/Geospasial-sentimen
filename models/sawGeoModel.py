import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# Pastikan path ini sesuai dengan folder Anda
GEOJSON_FILE = os.path.join(ROOT_DIR, "hotosm_idn_points_of_interest_points_geojson", "hotosm_idn_points_of_interest_points_geojson.geojson")

class GeoTOPSISPredictor:
    def __init__(self):
        print("--- Inisialisasi Geo Engine (Dynamic) ---")
        print(f"Loading data: {GEOJSON_FILE}...")
        
        # 1. Load Data Sekali Saja
        self.gdf = gpd.read_file(GEOJSON_FILE)
        self.gdf = self.gdf.to_crs(epsg=3857) # Ubah ke Meter
        
        # Buat Spatial Index agar pencarian cepat
        self.sindex = self.gdf.sindex
        print("Data Loaded. Siap menerima request dinamis.")

    def _get_neighbors_subset(self, point_geometry, radius=5000):
        """Mengambil data dalam radius 5KM (buffer kasar) untuk dipersempit nanti"""
        buffer = point_geometry.buffer(radius)
        possible_matches_index = list(self.sindex.query(buffer, predicate='intersects'))
        
        if not possible_matches_index:
            return gpd.GeoDataFrame()
            
        return self.gdf.iloc[possible_matches_index]

    def predict_score(self, lat, lon, competitor_tags, support_tags):
        """
        PERBAIKAN: Sekarang menerima 4 argumen (self, lat, lon, competitor, support).
        """
        # 1. Konversi Koordinat
        pt_geo = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
        
        # 2. Ambil data sekitar (Radius 5KM) 
        subset_gdf = self._get_neighbors_subset(pt_geo, radius=5000)
        
        if subset_gdf.empty:
            return 0.0

        
        def filter_by_tags(df, tags_list):
            if df.empty: return df
            mask = pd.Series([False] * len(df), index=df.index)
            
            # Cek kolom 'amenity' (Resto, Sekolah, Bank)
            if 'amenity' in df.columns:
                mask |= df['amenity'].isin(tags_list)
            # Cek kolom 'shop' (Bengkel, Minimarket)
            if 'shop' in df.columns:
                mask |= df['shop'].isin(tags_list)
            return df[mask]

        # 3. FILTER DATA BERDASARKAN TAGS DARI APP.PY
        competitors = filter_by_tags(subset_gdf, competitor_tags)
        supports = filter_by_tags(subset_gdf, support_tags)
        
        # 4. HITUNG FITUR
        
        # A. Support Density (Radius 500m)
        supports_500m = supports[supports.geometry.distance(pt_geo) <= 500]
        val_support = len(supports_500m)
        
        # B. Competitor Distance (Nearest Neighbor)
        if competitors.empty:
            val_dist = 5000 # Tidak ada saingan = Bagus (Jauh)
        else:
            dists = competitors.geometry.distance(pt_geo)
            val_dist = dists.min()
            if val_dist < 1: val_dist = 1 # Hindari 0

        # 5. NORMALISASI TOPSIS (Fixed Threshold)
        # Kita pakai angka patokan agar cepat
        MAX_SUPPORT_IDEAL = 50.0  # >50 gedung = Max Score
        MAX_DIST_IDEAL = 3000.0   # >3km jarak saingan = Max Score
        
        # Norm Support (Benefit)
        norm_support = val_support / MAX_SUPPORT_IDEAL
        norm_support = np.clip(norm_support, 0, 1)
        
        # Norm Distance (Benefit: Makin jauh makin bagus)
        norm_dist = val_dist / MAX_DIST_IDEAL
        norm_dist = np.clip(norm_dist, 0, 1)
        
        # 6. HITUNG SKOR
        # Bobot: Support 60%, Jarak Saingan 40%
        W_SUPP = 0.6
        W_DIST = 0.4
        
        y_supp = norm_support * W_SUPP
        y_dist = norm_dist * W_DIST
        
        # Ideal Points
        ideal_plus_supp = 1 * W_SUPP
        ideal_plus_dist = 1 * W_DIST
        
        ideal_min_supp = 0
        ideal_min_dist = 0
        
        # Jarak Euclidean
        d_plus = np.sqrt((y_supp - ideal_plus_supp)**2 + (y_dist - ideal_plus_dist)**2)
        d_min  = np.sqrt((y_supp - ideal_min_supp)**2 + (y_dist - ideal_min_dist)**2)
        
        if (d_min + d_plus) == 0:
            return 0.0
            
        final_score = d_min / (d_min + d_plus)
        
        return {
            "score": round(final_score, 4),
            "raw_data":{
                "support_count": int(val_support),
                "competitor_dist": float(val_dist)
            }

        }