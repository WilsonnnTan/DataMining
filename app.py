import csv
from utils.visualize import cluster_data
from flask import Flask, render_template, request
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
                    # Proses clustering
                    csv_info = cluster_data(file_path=file)
                    return render_template("cluster.html", csv_info=csv_info)
        else:
            csv_info = {"error": "File bukan CSV!"}
            return render_template("index.html", csv_info=csv_info)

if __name__ == "__main__":
    app.run(debug=True)

