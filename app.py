# app.py
import os
import sys
from flask import Flask, render_template, request
import joblib

# Setup path untuk Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

# Load model
model_path = os.path.join(BASE_DIR, 'model_used_cars.pkl')
model = joblib.load(model_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    milage_input = None

    if request.method == 'POST':
        try:
            milage_input = float(request.form['milage'])
            prediction = model.predict([[milage_input]])[0]
            prediction = round(float(prediction), 2)
        except (ValueError, KeyError):
            prediction = None

    return render_template(
        'index.html',
        prediction=prediction,
        milage_input=milage_input
    )


@app.route('/about')
def about():
    return render_template('about.html')


# Hanya jalankan app.run() saat testing lokal, BUKAN di Vercel
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))