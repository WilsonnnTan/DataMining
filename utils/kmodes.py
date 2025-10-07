import random


def matching_dissimilarity(row1, row2):
    """Hitung jarak (dissimilarity) antar dua baris kategorikal."""
    jarak = 0
    for a, b in zip(row1, row2):
        if a != b:
            jarak += 1
    return jarak

def get_mode_per_column(cluster):
    """Mode per kolom dalam cluster."""
    if not cluster:
        return None
    num_cols = len(cluster[0])
    modes = []
    for col_idx in range(num_cols):
        col_values = [row[col_idx] for row in cluster]
        # pilih nilai yang paling sering muncul
        mode_val = max(set(col_values), key=col_values.count)
        modes.append(mode_val)
    return modes

def kmodes(rows, k=2, max_iter=10000):
    """K-Modes untuk data kategorikal."""
    random.seed(2)
    centroids = random.sample(rows, k)

    for iteration in range(max_iter):
        clusters = [[] for _ in range(k)]

        # Assign
        for row in rows:
            # hitung jarak ke semua centroid
            distances = [matching_dissimilarity(row, c) for c in centroids]
            # ambil index centroid terdekat
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(row)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append(get_mode_per_column(cluster))
            else:
                new_centroids.append(random.choice(rows))

        # Cek konvergensi
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return clusters, centroids

def kmodes_cost(clusters, centroids):
    """Total dissimilarity (cost) K-Modes."""
    total_cost = 0
    for i, cluster in enumerate(clusters):
        for row in cluster:
            total_cost += matching_dissimilarity(row, centroids[i])
    return total_cost

