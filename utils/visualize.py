import utils.process as helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64


def run_process(file_path):
    # load data
    data = helper.load_csv(file_path)
    
    # Preprocessing: Scaling dan PCA
    print("Melakukan Preprocessing...")
    data_pca = helper.pca_from_scratch(data, n_components=2)
    print("Preprocessing Selesai.")

    # Menjalankan Algoritma Clustering
    print("\nMenjalankan Clustering...")
    k = 4
    kmeans_labels_fs = helper.kmeans_from_scratch(data_pca, k=k, random_state=42)
    ahc_labels_fs = helper.ahc_from_scratch(data_pca, n_clusters=k)
    dbscan_labels_fs = helper.dbscan_from_scratch(data_pca, eps=0.7, min_samples=4)
    print("Clustering Selesai.")

    # Evaluasi Skor
    print("\nMenghitung Skor...")
    kmeans_score_fs = helper.silhouette_score_from_scratch(data_pca, kmeans_labels_fs)
    ahc_score_fs = helper.silhouette_score_from_scratch(data_pca, ahc_labels_fs)

    # Untuk DBSCAN, abaikan noise points (label -1) saat menghitung skor

    # list kosong untuk menampung data non-noise
    core_data_dbscan = []
    core_labels_dbscan = []

    for i in range(len(dbscan_labels_fs)):
        # Jika labelnya BUKAN -1 (bukan noise)
        if dbscan_labels_fs[i] != -1:
            # Masukkan data dan labelnya ke list baru
            core_data_dbscan.append(data_pca[i])
            core_labels_dbscan.append(dbscan_labels_fs[i])

    # Ubah kembali ke NumPy array agar bisa dihitung
    core_data_dbscan = np.array(core_data_dbscan)

    # Hitung skor hanya pada data yang sudah difilter (non-noise)
    if len(core_data_dbscan) > 1 and len(set(core_labels_dbscan)) > 1:
        dbscan_score_fs = helper.silhouette_score_from_scratch(core_data_dbscan, core_labels_dbscan)
    else:
        dbscan_score_fs = float('nan') # Tidak dihitung jika tidak ada cluster

    print(f"Silhouette Score (from scratch) untuk K-Means: {kmeans_score_fs:.3f}")
    print(f"Silhouette Score (from scratch) untuk AHC: {ahc_score_fs:.3f}")
    print(f"Silhouette Score (from scratch) untuk DBSCAN: {dbscan_score_fs:.3f}")
    
    df_plot = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2'])
    df_plot['KMeans'] = kmeans_labels_fs
    df_plot['AHC'] = ahc_labels_fs
    df_plot['DBSCAN'] = dbscan_labels_fs

    # Fungsi bantu untuk buat plot base64
    def make_plot(data, x, y, hue, title, palette):
        plt.figure(figsize=(7, 6))
        sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=palette, s=100, alpha=0.85, edgecolor="black")
        plt.title(title, fontsize=13, fontweight='bold')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        return img_base64

    # Buat 3 plot terpisah
    plots = [
        {
            "title": f"K-Means (Skor: {kmeans_score_fs:.3f})",
            "image": make_plot(df_plot, "PCA1", "PCA2", "KMeans", "K-Means Clustering", "viridis")
        },
        {
            "title": f"DBSCAN (Skor: {dbscan_score_fs:.3f})",
            "image": make_plot(df_plot, "PCA1", "PCA2", "DBSCAN", "DBSCAN Clustering", "deep")
        },
        {
            "title": f"AHC (Skor: {ahc_score_fs:.3f})",
            "image": make_plot(df_plot, "PCA1", "PCA2", "AHC", "AHC Clustering", "plasma")
        }
    ]

    # Return hasil untuk template
    return {
        "kmeans_score": round(kmeans_score_fs, 3),
        "ahc_score": round(ahc_score_fs, 3),
        "dbscan_score": round(dbscan_score_fs, 3),
        "plots": plots
    }