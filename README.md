# Deteksi Uang Kertas Indonesia

Aplikasi Streamlit untuk mendeteksi nominal uang kertas Indonesia dari gambar menggunakan model Keras yang telah dipatch.

## Deskripsi

Proyek ini menampilkan antarmuka web ringan yang memungkinkan pengguna mengunggah foto uang kertas Indonesia, kemudian melakukan prediksi kelas nominal uang dengan model `model_patched.keras`.

Ada dua komponen utama:
- `patch_model.py`: script untuk mempatch model asli `model_hybrid_terbaik.keras` agar dapat dimuat dengan benar.
- `app.py`: aplikasi Streamlit yang menampilkan UI, memuat model, dan mengklasifikasikan gambar uang.

## Fitur

- Upload gambar uang kertas (JPG, JPEG, PNG, WEBP)
- Prediksi nominal uang kertas Indonesia untuk 8 kelas:
  - Rp 1.000
  - Rp 2.000
  - Rp 5.000
  - Rp 10.000
  - Rp 20.000
  - Rp 50.000
  - Rp 75.000
  - Rp 100.000
- Menampilkan confidence score
- Visualisasi probabilitas untuk setiap kelas
- Riwayat prediksi selama sesi Streamlit
- Status confidence dengan indikator aman / rendah

## Struktur Proyek

- `app.py` — aplikasi Streamlit utama
- `patch_model.py` — patch model Keras
- `model_hybrid_terbaik.keras` — model asli yang butuh patch
- `model_patched.keras` — model hasil patch yang siap digunakan
- `requirements.txt` — dependensi Python
- `.gitignore` — file dan folder yang diabaikan Git
- `.streamlit/` — konfigurasi Streamlit (jika ada)
- `app/` — aset tambahan, termasuk `model_hybrid_terbaik.keras.zip`

## Persyaratan

- Python 3.11 atau lebih baru
- Virtual environment (direkomendasikan)
- Library Python:
  - `streamlit`
  - `tensorflow`
  - `numpy`
  - `Pillow`
  - `matplotlib`

## Instalasi

1. Aktifkan virtual environment:

```bash
python -m venv venv
venv\Scripts\Activate.ps1    # PowerShell
# atau
venv\Scripts\activate.bat    # CMD
```

2. Instal dependensi:

```bash
pip install -r requirements.txt
```

## Menjalankan Proyek

1. Patch model Keras sekali saja:

```bash
python patch_model.py
```

2. Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

3. Buka URL yang ditampilkan di terminal (biasanya `http://localhost:8501`).

## Catatan Penting

- `app.py` memuat `model_patched.keras`.
- Jika `model_patched.keras` belum tersedia, jalankan `python patch_model.py` terlebih dahulu.
- Pastikan gambar yang diunggah jelas dan fokus pada uang kertas agar prediksi lebih akurat.

## Pengembangan

- Jika model asli berubah, sesuaikan `patch_model.py` untuk memperbarui patch Lambda layer atau konfigurasi model.
- Model diatur pada input 224×224 RGB.
- Aplikasi menggunakan caching resource Streamlit untuk memuat model sekali per sesi.

## Lisensi

Gunakan proyek ini untuk pembelajaran atau demonstrasi. Sesuaikan lisensi sesuai kebutuhan organisasi atau penelitian.
