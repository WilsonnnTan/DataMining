# from utils.preprocess import load_csv, encode_kategori
# from utils.kmodes import kmodes, kmodes_cost
# from utils.column_analysis import find_non_informative_columns, print_column_distribution_kmodes

# # === Load & preprocess data ===
# header, rows = load_csv("dataset.csv")
# header, rows = encode_kategori(header, rows)

# # Pastikan semua data int
# X = [list(map(int, row)) for row in rows]

# # =========================
# # K-Modes
# # =========================
# k = 2
# clusters_kmodes, centroids_kmodes = kmodes(X, k=k)
# cost_kmodes = kmodes_cost(clusters_kmodes, centroids_kmodes)

# print("\n=== K-Modes Results ===")
# for idx, cluster in enumerate(clusters_kmodes):
#     print(f"Cluster {idx} ({len(cluster)} data)")

# print(f"Total dissimilarity (cost): {cost_kmodes}")

# # =========================
# # Analisis kolom
# # =========================
# non_info_cols = find_non_informative_columns(clusters_kmodes, centroids_kmodes, header)
# print("\nKolom yang kurang membedakan cluster (bisa dibuang):")
# print(non_info_cols)

# print("\nDistribusi nilai tiap kolom per cluster:")
# print_column_distribution_kmodes(clusters_kmodes, header)

from flask import Flask, render_template, request
import csv
from io import StringIO

app = Flask(__name__)

# Baca rules CSV (header yang valid)
with open("csv_rules.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    valid_header = [h.strip() for h in next(reader)]  # Ambil header

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/cluster", methods=["POST"])
def cluster():
    csv_info = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        if file and file.filename.endswith(".csv"):
            # Baca CSV
            stream = StringIO(file.stream.read().decode("utf-8"))
            reader = csv.reader(stream)
            data = list(reader)

            if not data:
                csv_info = {"error": "CSV kosong!"}
                return render_template("index.html", csv_info=csv_info)
            else:
                header = [h.strip() for h in data[0]]
                rows = data[1:]

                # Validasi header
                if header != valid_header:
                    csv_info = {
                        "error": f"Header CSV tidak sesuai!"
                    }
                    return render_template("index.html", csv_info=csv_info)
                else:
                    # TODO: manggil fungsi cluster dan visualisasi
                    return render_template("cluster.html", csv_info=csv_info)
        else:
            csv_info = {"error": "File bukan CSV!"}
            return render_template("index.html", csv_info=csv_info)

if __name__ == "__main__":
    app.run(debug=True)

