# Prediksi Harga Mobil Bekas - TA-2 Regresi Linear

**Tugas Akhir (TA-2) Modul 02 - Algoritma Regresi Linear**
Mata Praktikum: Pembelajaran Mesin

Aplikasi web prediktif untuk memprediksi harga mobil bekas berdasarkan total jarak tempuh (milage) menggunakan algoritma Simple Linear Regression. Dibangun dengan Flask, Scikit-Learn, dan Bootstrap 5.

---

## Pendahuluan

Industri mobil bekas telah berkembang pesat seiring dengan meningkatnya kebutuhan masyarakat akan kendaraan pribadi yang terjangkau. Salah satu tantangan utama dalam transaksi mobil bekas adalah menentukan harga jual yang adil, baik dari sisi penjual maupun pembeli. Harga mobil bekas dipengaruhi oleh banyak faktor, namun variabel yang paling intuitif dan mudah diukur adalah jarak tempuh (milage). Mobil yang telah dipakai dalam jarak tempuh yang lebih jauh cenderung memiliki harga jual lebih rendah karena keausan komponen mesin dan risiko kerusakan yang lebih tinggi.

Regresi Linear Sederhana merupakan algoritma machine learning fundamental yang dapat memodelkan hubungan linear antara satu variabel independen (milage) dengan satu variabel dependen (price). Meskipun sederhana, algoritma ini memberikan interpretabilitas yang tinggi - kita dapat mengetahui secara langsung seberapa besar pengaruh setiap tambahan milage terhadap penurunan harga mobil.

---

## Deskripsi Dataset

Dataset yang digunakan adalah **used_cars.csv** yang berisi informasi mobil bekas di pasar Amerika Serikat. Dataset ini memiliki 4.009 baris dan 12 kolom, mencakup berbagai merek mulai dari Ford, Hyundai, Lexus, hingga merek premium seperti Audi dan Mercedes-Benz.

### Struktur Dataset

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| brand | object | Merek mobil (Ford, Audi, dll.) |
| model | object | Model mobil |
| model_year | int64 | Tahun produksi (1974-2024) |
| milage | object* | Jarak tempuh dalam miles (perlu cleaning) |
| fuel_type | object | Jenis bahan bakar (170 missing) |
| engine | object | Spesifikasi mesin |
| transmission | object | Jenis transmisi |
| ext_col | object | Warna eksterior |
| int_col | object | Warna interior |
| accident | object | Riwayat kecelakaan (113 missing) |
| clean_title | object | Status kepemilikan (596 missing) |
| price | object* | Harga jual dalam USD (perlu cleaning) |

*Kolom milage dan price awalnya bertipe string (contoh: "51,000 mi." dan "$10,300"), sehingga perlu dibersihkan menjadi numerik sebelum digunakan.

### Justifikasi Pemilihan Variabel

Karena tugas TA-2 mensyaratkan penggunaan Regresi Linear Sederhana (satu variabel X dan satu variabel y), dipilih milage sebagai variabel independen dan price sebagai variabel dependen. Pilihan ini didukung oleh:

- Intuisi pasar yang kuat: semakin tinggi milage, semakin rendah harga jual mobil.
- Korelasi Pearson awal sebesar -0,31 menunjukkan hubungan negatif yang signifikan.
- Kedua kolom bersifat numerik kontinu, sesuai dengan asumsi Regresi Linear.
- Milage adalah informasi yang mudah didapat oleh calon pembeli, sehingga model memiliki nilai praktis tinggi.

---

## Metodologi

Alur kerja data science yang diterapkan mengikuti tahapan standar: data loading, cleaning, EDA, splitting, training, evaluation, dan model persistence.

### Langkah 1 - Load Data

Dataset dimuat menggunakan Pandas. Pada tahap ini diperiksa tipe data setiap kolom dan jumlah missing values untuk merencanakan strategi cleaning.

```python
import pandas as pd
df = pd.read_csv("used_cars.csv")
print(df.shape)  # (4009, 12)
print(df.dtypes)
print(df.isnull().sum())
```

### Langkah 2 - Data Cleaning

Tahap paling kritis pada dataset ini adalah membersihkan kolom milage dan price yang masih berupa string. Selain itu, outlier ekstrem (mobil mewah dengan harga di atas $150.000 atau milage di atas 300.000) perlu dihapus agar tidak mengganggu pola linear mayoritas data.

```python
# Bersihkan milage: "51,000 mi." -> 51000
df["milage"] = df["milage"].str.replace(" mi.", "", regex=False)
df["milage"] = df["milage"].str.replace(",", "", regex=False)
df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

# Bersihkan price: "$10,300" -> 10300
df["price"] = df["price"].str.replace("$", "", regex=False)
df["price"] = df["price"].str.replace(",", "", regex=False)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Drop missing values
df = df.dropna(subset=["milage", "price"]).copy()

# Filter outlier ekstrem
df = df[(df["price"] <= 150000) & (df["milage"] <= 300000)]
print(df.shape)  # (3886, 12)
```

### Langkah 3 - Exploratory Data Analysis (EDA)

EDA bertujuan memvalidasi secara visual bahwa hubungan antara milage dan price cenderung linear. Scatterplot menunjukkan tren negatif: sebaran titik menurun dari kiri-atas ke kanan-bawah.

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(df["milage"], df["price"], alpha=0.3, s=10)
axes[0].set_xlabel("Milage (miles)")
axes[0].set_ylabel("Price (USD)")
axes[1].hist(df["price"], bins=50, color="#A23B72")
plt.savefig("eda_plot.png", dpi=150)

# Korelasi
print(df["milage"].corr(df["price"]))  # -0.31
```

### Langkah 4 - Train/Test Split

Data dibagi menjadi 80% training (3.107 baris) dan 20% testing (779 baris) menggunakan random_state=42 untuk reproducibility.

```python
from sklearn.model_selection import train_test_split

X = df[["milage"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Langkah 5 - Training Model

Model dilatih menggunakan LinearRegression dari scikit-learn. Proses training memanfaatkan metode Ordinary Least Squares (OLS) untuk mencari garis lurus terbaik yang meminimalkan jumlah squared error.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_:.4f}")
print(f"Koefisien: {model.coef_[0]:.4f}")
```

### Langkah 6 - Prediksi dan Evaluasi

Evaluasi dilakukan menggunakan empat metrik standar: MAE, MSE, RMSE, dan R-squared. Selain itu, visualisasi actual vs predicted membantu menilai kualitas model secara intuitif.

```python
from sklearn import metrics
import numpy as np

y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print(f"MAE : ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2  : {r2:.4f}")
```

### Langkah 7 - Simpan Model

Model yang telah dilatih disimpan ke file .pkl menggunakan joblib, sehingga dapat dimuat oleh aplikasi Flask tanpa perlu training ulang.

```python
import joblib
joblib.dump(model, "model_used_cars.pkl")

# Verifikasi
loaded = joblib.load("model_used_cars.pkl")
print(loaded.predict([[50000]]))  # [41229.32]
```

---

## Hasil dan Analisis

### Parameter Model

| Parameter | Nilai | Interpretasi |
|-----------|-------|--------------|
| Intercept (beta_0) | 56.720,39 | Harga dasar teoritis mobil saat milage = 0. Tidak realistis secara praktis, tetapi penting secara matematis sebagai titik potong garis regresi. |
| Koefisien milage (beta_1) | -0,3098 | Setiap tambahan 1 mile menurunkan harga sekitar $0,31. Setiap 10.000 mile, harga turun sekitar $3.098. |

### Persamaan Regresi

```
price = 56720.39 + (-0.3098) * milage
```

### Metrik Evaluasi

| Metrik | Nilai | Penjelasan |
|--------|-------|------------|
| MAE | $15.341,41 | Rata-rata selisih absolut antara prediksi dan aktual. Prediksi model rata-rata meleset sekitar $15.341 dari harga sebenarnya. |
| MSE | 4,2 x 10^8 | Mean Squared Error. Memberi bobot lebih besar pada error besar karena di-kuadratkan. |
| RMSE | $20.498,00 | Akar dari MSE, kembali ke satuan USD. Lebih sensitif terhadap outlier dibanding MAE. |
| R-squared | 0,4073 (40,73%) | Proporsi variasi harga yang dapat dijelaskan oleh milage. Sisanya (59,27%) dipengaruhi faktor lain. |

### Interpretasi Bisnis

Hasil model menunjukkan bahwa setiap tambahan 10.000 miles, harga mobil bekas turun rata-rata $3.098. Penurunan ini sesuai dengan intuisi pasar: mobil dengan kilometer tempuh tinggi mengalami keausan mesin, komponen, dan sistem kelistrikan yang lebih besar, sehingga nilai jualnya menurun. Bagi calon pembeli, model ini dapat digunakan sebagai panduan awal untuk menilai apakah harga yang ditawarkan penjual wajar atau tidak berdasarkan milage kendaraan.

Namun, nilai R-squared sebesar 40,73% mengindikasikan bahwa milage hanya menjelaskan sebagian variasi harga. Sisanya sebesar 59,27% dipengaruhi faktor lain yang tidak dimasukkan dalam model, antara lain: merek (brand), tahun produksi (model_year), jenis bahan bakar, riwayat kecelakaan, kondisi mesin, dan warna eksterior. Untuk meningkatkan akurasi, model dapat diperluas menjadi Multiple Linear Regression dengan menambahkan variabel-variabel tersebut.

---

## Implementasi Aplikasi Web Flask

### Struktur Folder

```
flask_app/
├── app.py                  # Backend Flask
├── model_used_cars.pkl     # Model yang sudah dilatih
├── train_model.py          # Script training (reproducibility)
├── used_cars.csv           # Dataset asli
├── used_cars_cleaned.csv   # Dataset hasil cleaning
├── requirements.txt        # Dependencies
├── Procfile                # Config deployment
├── runtime.txt             # Versi Python
├── templates/
│   ├── index.html          # Form prediksi
│   └── about.html          # Halaman info
└── static/
    └── style.css           # Styling
```

### Backend (app.py)

Backend Flask memuat model .pkl saat startup, lalu menerima input milage dari form HTML dan mengembalikan prediksi harga.

```python
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("model_used_cars.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        milage = float(request.form["milage"])
        prediction = model.predict([[milage]])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
```

### Frontend (index.html)

Antarmuka menggunakan Bootstrap 5 dengan form input milage, tombol prediksi, dan kartu hasil yang menampilkan estimasi harga. Halaman about.html menjelaskan metodologi dan interpretasi model.

---

## Cara Menjalankan Secara Lokal

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/carprice-predictor.git
cd carprice-predictor
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan Model

Pastikan file `model_used_cars.pkl` sudah ada. Jika belum, jalankan:

```bash
python train_model.py
```

### 5. Jalankan Aplikasi

```bash
python app.py
```

Buka browser ke: http://127.0.0.1:5000

---

## Deployment ke Render

1. Push seluruh folder ini ke repository GitHub (public).
2. Buka https://render.com dan login dengan GitHub.
3. Pilih "New Web Service" lalu hubungkan ke repository GitHub Anda.
4. Set Build Command: `pip install -r requirements.txt`
5. Set Start Command: `gunicorn app:app`
6. Klik "Create Web Service" dan tunggu hingga build selesai.
7. Salin URL aplikasi untuk dimasukkan ke laporan.

---

## Deployment ke Railway

1. Buka https://railway.app dan login dengan GitHub.
2. New Project -> Deploy from GitHub repo.
3. Railway akan otomatis mendeteksi Procfile.
4. Tunggu hingga build selesai dan salin URL aplikasi.

---

## Requirements

```
Flask==3.0.0
scikit-learn==1.3.2
joblib==1.3.2
numpy==1.26.2
pandas==2.1.4
gunicorn==21.2.0
```

Python version: 3.11.7 (lihat file runtime.txt)

---

## Kesimpulan

Pada tugas akhir ini, berhasil diimplementasikan algoritma Regresi Linear Sederhana untuk memprediksi harga mobil bekas berdasarkan milage. Model menghasilkan persamaan `price = 56720.39 - 0.3098 * milage` dengan R-squared = 40,73% dan MAE = $15.341. Hasil ini menunjukkan bahwa milage memiliki pengaruh negatif yang signifikan terhadap harga mobil bekas - setiap tambahan 10.000 miles menurunkan harga rata-rata $3.098.

Model kemudian dikemas dalam aplikasi web interaktif menggunakan Flask dengan antarmuka Bootstrap yang responsif. Aplikasi memungkinkan pengguna memasukkan nilai milage dan mendapatkan estimasi harga secara real-time. Aplikasi telah dideploy ke platform cloud (Render/Railway) sehingga dapat diakses publik sebagai portofolio proyek.

Keterbatasan utama model adalah R-squared yang relatif rendah (40,73%), yang menunjukkan bahwa milage saja tidak cukup untuk menjelaskan seluruh variasi harga. Untuk pengembangan selanjutnya, model dapat diperluas menjadi Multiple Linear Regression dengan menambahkan variabel brand, tahun produksi, dan riwayat kecelakaan. Selain itu, algoritma non-linear seperti Random Forest atau Gradient Boosting dapat dicoba untuk menangani pola yang lebih kompleks.

---

## Referensi

1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning. Springer.
2. Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis. Wiley.
3. Scikit-learn Documentation: LinearRegression. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
4. Flask Documentation. https://flask.palletsprojects.com/
5. Kaggle: Used Cars Dataset. https://www.kaggle.com/datasets/

---

## Author

- Nama: [NAMA ANDA]
- NIM: [NIM ANDA]
- Mata Praktikum: Pembelajaran Mesin
- Modul: 02 - Algoritma Regresi Linear
- Tugas: TA-2: Implementasi Regresi Linear dengan Flask

---

## Acknowledgments

- Dosen Pengampu: Mohammad Bayu Anggara, S.Kom., M.Kom.
- Dataset: Kaggle Used Cars Dataset
- Inspirasi: Komunitas Data Science Indonesia
