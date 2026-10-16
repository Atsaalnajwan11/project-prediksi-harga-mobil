"""
=============================================================
CARFORGE - Aplikasi Web Prediksi Harga Mobil Bekas
Vercel-Ready Version
=============================================================
"""
import os
import sys
import logging
import traceback
from functools import lru_cache
from flask import Flask, render_template, request, jsonify

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Vercel fallback path
if not os.path.exists(os.path.join(BASE_DIR, 'model_used_cars.pkl')):
    alt_paths = [
        os.environ.get('LAMBDA_TASK_ROOT', ''),
        '/var/task',
        os.getcwd()
    ]
    for path in alt_paths:
        if path and os.path.exists(os.path.join(path, 'model_used_cars.pkl')):
            BASE_DIR = path
            break

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# =========================================================
# LAZY LOAD MODEL (dengan cache)
# =========================================================
MODEL_PATH = os.path.join(BASE_DIR, 'model_used_cars.pkl')
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"File exists: {os.path.exists(MODEL_PATH)}")

@lru_cache(maxsize=1)
def get_model():
    """Lazy load model, cache di memory"""
    try:
        import joblib
        logger.info("Loading model...")
        m = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded!")
        return m
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        logger.error(traceback.format_exc())
        return None


# =========================================================
# ROUTE: HOME
# =========================================================
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Index error: {e}")
        return str(e), 500


# =========================================================
# ROUTE: PREDIKSI
# =========================================================
@app.route('/prediksi', methods=['GET', 'POST'])
def predict():
    """Halaman prediksi"""
    prediction = None
    raw_value = None
    milage = None
    error = None
    trace = []

    if request.method == 'POST':
        try:
            raw_value = request.form.get('milage', '').strip()
            trace.append(('1. RAW INPUT', raw_value or '(empty)'))

            if not raw_value:
                error = "Input tidak boleh kosong."
            else:
                cleaned = raw_value.replace(' mi.', '').replace(',', '').strip()
                trace.append(('2. CLEANING', f'"{raw_value}" -> "{cleaned}"'))

                try:
                    milage = float(cleaned)
                    trace.append(('3. PARSE FLOAT', str(milage)))
                except ValueError:
                    error = f'"{raw_value}" bukan angka valid.'

                if not error and milage is not None:
                    if milage < 0:
                        error = "Milage tidak boleh negatif."
                        trace.append(('4. VALIDASI', 'FAIL: negative'))
                    elif milage > 300000:
                        error = f"Milage {milage:,.0f} melebihi batas 300.000 miles."
                        trace.append(('4. VALIDASI', 'FAIL: outlier'))
                    else:
                        trace.append(('4. VALIDASI', 'PASS'))

                if not error and milage is not None:
                    model = get_model()
                    if model is not None:
                        import pandas as pd
                        input_df = pd.DataFrame({'milage': [milage]})
                        raw_pred = model.predict(input_df)[0]
                        prediction = "${:,.2f}".format(round(float(raw_pred), 2))
                        trace.append(('5. PREDICT', f'-> {raw_pred:.4f}'))
                        trace.append(('6. OUTPUT', prediction))
                    else:
                        error = "Model belum siap di server."
                        trace.append(('5. PREDICT', 'FAIL: no model'))

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
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"About error: {e}")
        return str(e), 500


# =========================================================
# ROUTE: HEALTH CHECK (debugging)
# =========================================================
@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'base_dir': BASE_DIR,
        'model_path': MODEL_PATH,
        'file_exists': os.path.exists(MODEL_PATH),
        'model_loaded': get_model() is not None
    })


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)