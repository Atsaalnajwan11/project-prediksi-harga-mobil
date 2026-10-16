"""
=================================================
Aplikasi Prediksi Harga Mobil Bekas
Tugas Akhir — Pembelajaran Mesin (Modul 02: Regresi Linear)
-------------------------------------------------
Model     : Simple Linear Regression — model_used_cars.pkl
Persamaan : price = 56720.39 + (−0.3098) × milage   (Laporan §4.1)
Alur      : input → cleaning → validasi → predict → output
            (mengikuti Laporan §3.2 & §5.2)
=================================================
"""

import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ------------------------------------------------
# 1. Load model saat startup (Laporan §5.2)
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_used_cars.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[STARTUP] Model berhasil dimuat: {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model tidak ditemukan di '{MODEL_PATH}'. "
        "Pastikan 'model_used_cars.pkl' ada di folder root proyek."
    )

# ------------------------------------------------
# Konstanta model (Laporan §3.2 & §4.1)
# ------------------------------------------------
INTERCEPT = 56720.39      # β0
COEF = -0.3098            # β1 (ditampilkan dibulatkan 4 desimal)
MILAGE_MIN = 0
MILAGE_MAX = 300_000      # filter outlier milage (Laporan §3.2)


# ------------------------------------------------
# Pipeline prediksi — sesuai laporan
# ------------------------------------------------
def clean_milage(raw: str) -> str:
    """Cleaning milage — replikasi Laporan §3.2: '51,000 mi.' -> '51000'"""
    cleaned = raw.lower().replace(" mi.", "").replace("mi.", "")
    cleaned = cleaned.replace(",", "").replace(" ", "").strip()
    return cleaned


def run_prediction_pipeline(raw: str):
    """
    Alur prediksi lengkap (sesuai laporan):
      1. Ambil input dari form         (§5.2)
      2. Cleaning milage               (§3.2)
      3. Validasi & filter outlier     (§3.2)
      4. model.predict([[milage]])[0]  (§5.2, terverifikasi di §3.7)
      5. Format output USD
    Return: (prediction_str, milage, error, trace)
    """
    trace = []

    # -- 1. Input mentah ------------------------------------------
    trace.append(("1 · Input mentah (form)", raw if raw else "(kosong)"))

    # -- 2. Cleaning (Laporan §3.2) --------------------------------
    cleaned = clean_milage(raw)
    trace.append(("2 · Cleaning '51,000 mi.' → 51000 (§3.2)",
                  cleaned if cleaned else "(kosong)"))

    if not cleaned:
        return None, None, "Input milage tidak boleh kosong.", trace

    # -- 3. Konversi numerik (pd.to_numeric, errors='coerce') ------
    try:
        milage = float(cleaned)
    except ValueError:
        return None, None, (
            "Input harus berupa angka. Contoh: 45000 — "
            "format '51,000 mi.' juga diterima (Laporan §3.2)."
        ), trace

    # -- 4. Validasi (filter outlier §3.2) --------------------------
    if milage < MILAGE_MIN:
        return None, None, "Milage tidak boleh negatif.", trace
    if milage > MILAGE_MAX:
        return None, None, (
            f"Milage {milage:,.0f} miles di luar rentang training model "
            f"(maks 300.000 miles — filter outlier, Laporan §3.2)."
        ), trace
    trace.append(("3 · Validasi 0 ≤ milage ≤ 300.000 (§3.2)", "PASS"))

    # -- 5. Prediksi (§5.2: model.predict([[milage]])[0]) -----------
    X_new = pd.DataFrame({"milage": [milage]})  # setara [[milage]],
    y_pred = float(model.predict(X_new)[0])     # menghindari warning feature-name
    trace.append(("4 · model.predict([[milage]])[0] (§5.2)", f"{y_pred:,.2f}"))

    # -- 6. Cross-check persamaan (§4.1) -----------------------------
    manual = INTERCEPT + COEF * milage
    trace.append(("5 · Persamaan β0 + β1 × x (§4.1)", f"{manual:,.2f}"))

    # -- 7. Format output ---------------------------------------------
    prediction = "${:,.2f}".format(y_pred)
    trace.append(("6 · Output $XX,XXX.XX", prediction))

    return prediction, milage, None, trace


# ------------------------------------------------
# Route "/" — halaman utama + form cepat (GET & POST)
# ------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = input_milage = error = None
    if request.method == "POST":
        raw = request.form.get("milage", "")
        prediction, input_milage, error, _ = run_prediction_pipeline(raw)
    return render_template("index.html",
                           prediction=prediction,
                           input_milage=input_milage,
                           error=error)


# ------------------------------------------------
# Route "/predict" — HALAMAN PREDIKSI + EXECUTION TRACE
# ------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = milage = error = None
    trace, raw_value = [], ""
    if request.method == "POST":
        raw_value = request.form.get("milage", "")
        prediction, milage, error, trace = run_prediction_pipeline(raw_value)
    return render_template("predict.html",
                           prediction=prediction,
                           milage=milage,
                           error=error,
                           trace=trace,
                           raw_value=raw_value)


# ------------------------------------------------
# Route "/about" — halaman info proyek
# ------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# ------------------------------------------------
# Run lokal — PORT dari environment (deployment)
# ------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)