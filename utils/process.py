import csv
import pandas as pd
import numpy as np


def load_csv(file_path):
    # --- 1.1 Memuat dan Membersihkan Data ---
    df = pd.read_csv(file_path)
    columns_to_drop = ['Timestamp', 'Jenis Kelamin', 'Angkatan (Tahun Masuk)', 'Semester saat ini', 'IPK terakhir', 'Status aktivitas akademik', 'Platform media sosial yang sering dipakai']
    df_cleaned = df.drop(columns=columns_to_drop)

    # --- 1.2 Encoding (One-Hot) ---
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

    # Memastikan semua data numerik
    df_numeric = df_encoded.astype(np.float64)

    # --- 1.3 Konversi ke NumPy Array ---
    data_initial = df_numeric.to_numpy()
    
    return data_initial


# def print_data(header, rows):
#     """
#     Mencetak informasi dataset.
#     """
#     print(f"Total feature: {len(header)}")
#     print(f"Total row: {len(rows)}\n")

#     for i in range(len(header)):
#         print(f"{i}: {header[i]}")

#     for i in range(len(header)):
#         print(f"{i},", end="")
#     print()
        
#     for i in range(len(rows)):
#         print(rows[i])
       
        
def check_missing_value(rows):
    """
    Melakukan pengecekan missing value pada dataset.
    """
    total_miss = 0
    for row in rows:
        for col_idx in range(len(row)):
            if row[col_idx] == "":
                total_miss += 1
    print(f"Total sel dengan missing value: {total_miss}")


def remove_missing_rows(rows):
    """
    Menghapus baris yang memiliki missing value ("").
    Mengembalikan dataset baru tanpa missing value.
    """
    cleaned_rows = []

    for row in rows:
        if "" not in row:
            cleaned_rows.append(row)
            
    return cleaned_rows


def check_duplicates(rows):
    """
    Melakukan pengecekan duplikasi data.
    """
    seen = set()
    duplicates = []

    for row in rows:
        row_tuple = tuple(row)
        if row_tuple in seen:
            duplicates.append(row)
        else:
            seen.add(row_tuple)

    # Tampilkan hasil
    if duplicates:
        print(f"\nDitemukan {len(duplicates)} baris duplikat:")
        for d in duplicates:
            print(d)
    else:
        print("\nTidak ada duplikasi data ditemukan.")
        

def remove_duplicates(rows):
    """
    Menghapus baris duplikat dari dataset.
    """
    unique_rows = []
    seen = set()

    for row in rows:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)

    return unique_rows


def standard_scaler_from_scratch(X):
    """Standarisasi data (mean=0, std=1) per kolom."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Tambahkan nilai kecil (epsilon) untuk menghindari pembagian dengan nol
    epsilon = 1e-8
    return (X - mean) / (std + epsilon)


def pca_from_scratch(X, n_components):
    """Principal Component Analysis dari awal."""
    # Standarisasi data terlebih dahulu
    X_scaled = standard_scaler_from_scratch(X)
    
    # Hitung matriks kovarians
    cov_matrix = np.cov(X_scaled.T)
    
    # Hitung eigenvectors dan eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Urutkan eigenvectors berdasarkan eigenvalues
    eigen_pairs = [0] * len(eigenvalues) 
    for i in range(len(eigenvalues)):
        eigen_pairs[i] = (np.abs(eigenvalues[i]), eigenvectors[:, i])
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Pilih komponen utama
    # Siapkan array NumPy kosong dengan ukuran yang tepat
    n_features = eigenvectors.shape[0]
    projection_matrix_temp = np.zeros((n_components, n_features))

    # Loop untuk mengisi setiap baris pada array
    for i in range(n_components):
        vector = np.array(eigen_pairs[i][1])
        projection_matrix_temp[i] = vector
    projection_matrix = projection_matrix_temp.T 

    projection_matrix = np.array([eigen_pairs[i][1] for i in range(n_components)]).T
    
    # Proyeksikan data
    return X_scaled.dot(projection_matrix)


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))


def kmeans_from_scratch(X, k, max_iters=100, random_state=42):
    np.random.seed(random_state)
    rand_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[rand_indices, :]
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for idx, point in enumerate(X):
            closest_centroid_idx = np.argmin([euclidean_distance(point, c) for c in centroids])
            clusters[closest_centroid_idx].append(idx)
        
        old_centroids = centroids.copy()
        for i, cluster in enumerate(clusters):
            if cluster: centroids[i] = np.mean(X[cluster], axis=0)
        
        if np.all([euclidean_distance(old_centroids[i], centroids[i]) for i in range(k)]) == 0:
            break
            
    labels = np.empty(X.shape[0], dtype=int)
    for i, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels[point_idx] = i
    return labels


def dbscan_from_scratch(X, eps, min_samples):
    n_points = X.shape[0]
    labels = -1 * np.ones(n_points) # -1 untuk noise
    cluster_id = 0

    for i in range(n_points):
        if labels[i] != -1: continue # Sudah diproses
        
        neighbors = [j for j in range(n_points) if euclidean_distance(X[i], X[j]) < eps]
        
        if len(neighbors) < min_samples:
            continue # Tetap sebagai noise untuk sementara
        
        # Titik ini adalah core point, mulai cluster baru
        labels[i] = cluster_id
        seed_set = set(neighbors)
        seed_set.remove(i)
        
        while seed_set:
            current_point = seed_set.pop()
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
            
            current_neighbors = [j for j in range(n_points) if euclidean_distance(X[current_point], X[j]) < eps]
            
            if len(current_neighbors) >= min_samples:
                new_neighbors = set(current_neighbors)
                # Tambahkan titik yang belum diproses ke seed_set
                seed_set.update([p for p in new_neighbors if labels[p] == -1])
        
        cluster_id += 1
    return labels


def ahc_from_scratch(X, n_clusters):
    # Menggunakan metode single-linkage (jarak minimum antar cluster)
    n_samples = X.shape[0]
    clusters = [[i] for i in range(n_samples)]
    
    while len(clusters) > n_clusters:
        min_dist = float('inf')
        merge_idx = (-1, -1)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Cari jarak minimum antar titik di dua cluster
                dist = min([euclidean_distance(X[p1], X[p2]) for p1 in clusters[i] for p2 in clusters[j]])
                if dist < min_dist:
                    min_dist = dist
                    merge_idx = (i, j)
        
        # Gabungkan dua cluster terdekat
        i, j = merge_idx
        clusters[i].extend(clusters[j])
        clusters.pop(j)
        
    labels = np.empty(n_samples, dtype=int)
    for i, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels[point_idx] = i
    return labels


def silhouette_score_from_scratch(X, labels):
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return float('nan') # Skor tidak bisa dihitung
    
    silhouette_vals = []
    for i in range(n_samples):
        # a(i): Rata-rata jarak ke titik lain di cluster yang sama
        own_cluster_label = labels[i]
        own_cluster_indices = np.where(labels == own_cluster_label)[0]
        if len(own_cluster_indices) <= 1:
            a_i = 0
        else:
            a_i = np.mean([euclidean_distance(X[i], X[j]) for j in own_cluster_indices if i != j])
        
        # b(i): Rata-rata jarak minimum ke cluster lain
        b_i = float('inf')
        for label in unique_labels:
            if label == own_cluster_label: continue
            other_cluster_indices = np.where(labels == label)[0]
            mean_dist = np.mean([euclidean_distance(X[i], X[j]) for j in other_cluster_indices])
            if mean_dist < b_i:
                b_i = mean_dist
        
        if max(a_i, b_i) == 0:
             s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        silhouette_vals.append(s_i)
    
    return np.mean(silhouette_vals)