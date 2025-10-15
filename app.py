import csv
from utils.visualize import run_process
from flask import Flask, render_template, request, send_file
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
    file = request.files.get("csv_file")

    if not file or not file.filename.endswith(".csv"):
        csv_info = {"error": "File bukan CSV!"}
        return render_template("index.html", csv_info=csv_info)

    stream = StringIO(file.stream.read().decode("utf-8"))
    reader = csv.reader(stream)
    data = list(reader)

    if not data:
        csv_info = {"error": "CSV kosong!"}
        return render_template("index.html", csv_info=csv_info)

    header = [h.strip() for h in data[0]]
    rows = data[1:]

    # Validasi header dengan csv_rules.csv
    if header != valid_header:
        csv_info = {"error": "Header CSV tidak sesuai!"}
        return render_template("index.html", csv_info=csv_info)

    # === Simpan hasil upload ke dataset.csv (overwrite) ===
    with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    csv_info = run_process("dataset.csv")
    return render_template("cluster.html", csv_info=csv_info)


@app.route("/download_question", methods=["GET"])
def download_question():
    return send_file(
        "list_pertanyaan.txt",
        as_attachment=True,
        download_name="list_pertanyaan.txt",
        mimetype="text/csv"
    )


if __name__ == "__main__":
    app.run(debug=True)
