"""
=============================================================
CARFORGE - Aplikasi Web Prediksi Harga Mobil Bekas
=============================================================
"""
import os
import sys
import logging
import traceback
from flask import Flask, render_template, request

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup path (penting untuk Vercel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# =========================================================
# LOAD MODEL DENGAN ERROR HANDLING
# =========================================================
MODEL_PATH = os.path.join(BASE_DIR, 'model_used_cars.pkl')
logger.info(f"Loading model from: {MODEL_PATH}")

model = None
try:
    import joblib
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None


# =========================================================
# ROUTE: HOME
# =========================================================
@app.route('/')
def index():
    return render_template('index.html')


# =========================================================
# ROUTE: PREDIKSI
# =========================================================
@app.route('/prediksi', methods=['GET', 'POST'])
def predict():
    """Halaman prediksi - sesuai Laporan §5.2"""
    prediction = None
    raw_value = None
    milage = None
    error = None
    trace = []

    if request.method == 'POST':
        try:
            # 1. Raw input
            raw_value = request.form.get('milage', '').strip()
            trace.append(('1. RAW INPUT', raw_value or '(empty)'))

            if not raw_value:
                error = "Input tidak boleh kosong."
            else:
                # 2. Cleaning
                cleaned = raw_value.replace(' mi.', '').replace(',', '').strip()
                trace.append(('2. CLEANING', f'"{raw_value}" -> "{cleaned}"'))

                try:
                    milage = float(cleaned)
                    trace.append(('3. PARSE FLOAT', f'{milage}'))
                except ValueError:
                    error = f'"{raw_value}" bukan angka yang valid.'

                # 3. Validasi
                if not error and milage is not None:
                    if milage < 0:
                        error = "Milage tidak boleh negatif."
                        trace.append(('4. VALIDASI', 'FAIL: negative'))
                    elif milage > 300000:
                        error = f"Milage {milage:,.0f} melebihi batas 300.000 miles (outlier)."
                        trace.append(('4. VALIDASI', 'FAIL: outlier'))
                    else:
                        trace.append(('4. VALIDASI', f'PASS: 0 <= {milage} <= 300000'))

                # 4. Prediksi
                if not error and milage is not None and model is not None:
                    import pandas as pd
                    input_df = pd.DataFrame({'milage': [milage]})
                    raw_pred = model.predict(input_df)[0]
                    prediction = "${:,.2f}".format(round(float(raw_pred), 2))
                    trace.append(('5. MODEL.PREDICT', f'-> {raw_pred:.4f}'))
                    trace.append(('6. OUTPUT FORMAT', prediction))

                elif model is None and not error:
                    error = "Model belum siap di server."
                    trace.append(('5. MODEL.PREDICT', 'FAIL: model not loaded'))

        except Exception as e:
            logger.error(f"Predict error: {e}")
            logger.error(traceback.format_exc())
            error = f"Terjadi kesalahan: {str(e)}"

    return render_template(
        'predict.html',
        prediction=prediction,
        raw_value=raw_value or '',
        milage=milage,
        error=error,
        trace=trace
    )


# =========================================================
# ROUTE: ABOUT
# =========================================================
@app.route('/about')
def about():
    return render_template('about.html')


# =========================================================
# ROUTE: HEALTH CHECK
# =========================================================
@app.route('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'file_exists': os.path.exists(MODEL_PATH)
    }


# =========================================================
# MAIN (untuk lokal)
# =========================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)