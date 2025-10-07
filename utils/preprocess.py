import csv
from collections import Counter


def load_csv(file_path):
    """
    Memuat file CSV dan melakukan pengurangan data.
    Menghapus kolom yang tidak diperlukan (Data Reduction).
    """
    file_path = "dataset.csv"
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)

    header = data[0]
    rows = data[1:]

    # Kolom yang mau dihapus (Timestamp, Jenis Kelamin, Angkatan, Semester, Status aktivitas, Platform medsos)
    drop_indices = [6, 5, 3, 2, 1, 0]

    for r in rows:
        for i in drop_indices:
            r.pop(i)

    for i in drop_indices:
        header.pop(i)

    header = [h.strip() for h in header]
    
    for r in rows:
        for i in range(len(r)):
            r[i] = r[i].strip()
        
    return header, rows


def print_data(header, rows):
    """
    Mencetak informasi dataset.
    """
    print(f"Total feature: {len(header)}")
    print(f"Total row: {len(rows)}\n")

    for i in range(len(header)):
        print(f"{i}: {header[i]}")

    for i in range(len(header)):
        print(f"{i},", end="")
    print()
        
    for i in range(len(rows)):
        print(rows[i])
       
        
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
        

def encode_kategori(header, rows):
    """
    Melakukan encoding pada kolom kategorikal.
    """
    target_col = [
        "IPK terakhir",
        "Rata-rata waktu penggunaan media sosial per hari",
        "Proporsi konten edukasi dari total waktu sosmed (persepsi)",
        "Ketika tugas sulit, langkah pertama saya",
        "Saya bersedia menggunakan platform campus-official yang mengkurasi konten sosial",
    ]
    for target in target_col:
        if target in header:
            target_index = header.index(target)
            
            unique_values = []
            for row in rows:
                value = row[target_index].strip()
                if value not in unique_values:
                    unique_values.append(value)
            
            kategori_mapping = {}
            for i, val in enumerate(unique_values):
                kategori_mapping[val] = i
                
            for row in rows:
                row[target_index] = kategori_mapping[row[target_index].strip()]
                
            print("\n=== Mapping kategori otomatis ===")
            for key, val in kategori_mapping.items():
                print(f"{val}: {key}")
        else:
            print(f"Kolom '{target}' tidak ditemukan di header!")
    
    return header, rows


def data_reduction(header_names, header, rows):
    """
    Melakukan pengurangan data (Data Reduction).
    """
    for header_name in header_names:
        if header_name in header:
            idx = header.index(header_name)
            for row in rows:
                row.pop(idx)
            header.pop(idx)
            print(f"Kolom '{header_name}' dihapus.")
        else:
            print(f"Kolom '{header_name}' tidak ditemukan di header!")
            

def find_non_informative_columns(clusters, centroids, header):
    """
    Kolom yang mode-nya sama di semua cluster → kurang membedakan
    """
    non_informative = []
    for col_idx in range(len(header)):
        values = [cent[col_idx] for cent in centroids]
        if len(set(values)) == 1:
            non_informative.append(header[col_idx])
    return non_informative


def print_column_distribution_kmodes(clusters, header):
    """
    Print distribusi nilai tiap kolom per cluster
    clusters: list of list of rows
    """
    num_cols = len(header)
    for col_idx in range(num_cols):
        print(f"\nKolom: {header[col_idx]}")
        for cluster_idx, cluster in enumerate(clusters):
            values = [row[col_idx] for row in cluster]
            counts = Counter(values)
            print(f" Cluster {cluster_idx}: {dict(counts)}")