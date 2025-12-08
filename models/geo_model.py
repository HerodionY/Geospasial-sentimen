# models/geo_model.py
import osmnx as ox
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.cluster import KMeans

# Simulasi data historis kepadatan (untuk training model)
# Fitur: [Kepadatan Penduduk, Kepadatan Fasilitas Publik, Jarak ke Jalan Utama]
HISTORICAL_DATA = np.array([
    [500, 10, 50], [450, 8, 60], [100, 2, 500],
    [900, 15, 20], [850, 12, 30], [200, 3, 400]
])

# Latih K-Means (misalnya 2 klaster: High Potential dan Low Potential)
KMEANS_MODEL = KMeans(n_clusters=2, random_state=42, n_init=10).fit(HISTORICAL_DATA)

def get_osm_features(lat, lon, competitor_tags, support_tags, radius=500):
    """
    Mengambil dan menghitung fitur Geo-spasial dari OSM.
    """
    try:
        # Mengambil POI dalam radius
        poi_competitor = ox.features.features_from_point((lat, lon), tags=competitor_tags, dist=radius)
        poi_support = ox.features.features_from_point((lat, lon), tags=support_tags, dist=radius)

        competitor_count = len(poi_competitor)
        support_count = len(poi_support)
        
        # Simulasi Jarak ke Pusat Kota (Gunakan Geopy Distance)
        pusat_kota_coord = (-6.175, 106.828) # Contoh koordinat Jakarta Pusat
        current_coord = (lat, lon)
        distance_to_center = geodesic(current_coord, pusat_kota_coord).km 
        
        return competitor_count, support_count, distance_to_center

    except Exception as e:
        print(f"Error OSMnx/Geopy: {e}")
        return 0, 0, 1000 # Nilai default buruk jika gagal

def run_geo_analysis(lat, lon):
    # Definisi Tags OSM (Sesuaikan dengan ritel Anda!)
    # Contoh: Jika ritel Anda adalah kafe/restoran:
    COMPETITOR_TAGS = {"amenity": "restaurant"}
    SUPPORT_TAGS = {"amenity": ["school", "university", "office"]}
    
    comp_count, supp_count, dist_center = get_osm_features(lat, lon, COMPETITOR_TAGS, SUPPORT_TAGS)
    
    current_features = np.array([[comp_count, supp_count, dist_center]])
    cluster = KMEANS_MODEL.predict(current_features)[0]
    
    if cluster == 2:
        c1_score = 0.9  # High Potential
    elif cluster == 1:
        c1_score = 0.65  # Low Potential
    else:
        c1_score = 0.40  # Netral

    c1_score += np.random.rand() * 0.05
    
    return round(c1_score, 3)