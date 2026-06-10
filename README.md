# Construction Safety Monitor — Capstone Project Module 4

Aplikasi deteksi kelengkapan APD (Alat Pelindung Diri) pekerja konstruksi menggunakan YOLOv8.

## Struktur Project

```
capstone4/
├── notebook/
│   └── Capstone4_ConstructionSafety.ipynb   ← Training di Google Colab
└── streamlit_app/
    ├── app.py                                ← Aplikasi Streamlit
    ├── requirements.txt                      ← Python dependencies
    ├── packages.txt                          ← System dependencies (Streamlit Cloud)
    ├── .streamlit/
    │   └── config.toml                       ← Konfigurasi Streamlit
    └── construction_safety_best.pt           ← Model (letakkan di sini setelah training)
```

---

## LANGKAH 1 — Training Model di Google Colab

1. Buka [Google Colab](https://colab.research.google.com)
2. Upload file `Capstone4_ConstructionSafety.ipynb`
3. Aktifkan GPU: **Runtime → Change runtime type → T4 GPU**
4. Upload dataset ke Google Drive sesuai struktur:
   ```
   MyDrive/capstone4/construction_safety/
   ├── train/ (images/ + labels/)
   ├── valid/ (images/ + labels/)
   └── test/  (images/ + labels/)
   ```
5. Jalankan semua cell secara berurutan (**Runtime → Run all**)
6. Tunggu training selesai (~1-3 jam)
7. Model tersimpan otomatis di `MyDrive/capstone4/model/construction_safety_best.pt`

---

## LANGKAH 2 — Testing di Lokal (Local Run)

### Prasyarat
- Python 3.9 atau lebih baru
- pip

### Instalasi

```bash
# 1. Clone atau download folder streamlit_app
cd streamlit_app

# 2. Buat virtual environment (sangat disarankan)
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Letakkan file model .pt di folder ini
# Salin construction_safety_best.pt ke streamlit_app/

# 5. Jalankan aplikasi
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## LANGKAH 3 — Deploy ke Streamlit Community Cloud

### 3a. Siapkan model untuk Cloud

File `.pt` tidak bisa diupload ke GitHub (terlalu besar).
Gunakan Google Drive sebagai model storage:

1. Buka Google Drive → cari `construction_safety_best.pt`
2. Klik kanan → **Share** → **Anyone with the link** → **Viewer**
3. Copy link. Ambil **FILE_ID** dari URL:
   ```
   https://drive.google.com/file/d/[FILE_ID_INI]/view
   ```
4. Buka `app.py`, edit baris berikut:
   ```python
   GDRIVE_FILE_ID = "paste_file_id_disini"
   ```

### 3b. Upload ke GitHub

```bash
# Inisialisasi git di folder streamlit_app
git init
git add app.py requirements.txt packages.txt .streamlit/ .gitignore
git commit -m "Capstone4 - Construction Safety Monitor"

# Buat repo baru di github.com (HARUS PUBLIC)
git remote add origin https://github.com/username/capstone4-safety.git
git push -u origin main
```

> **Penting:** Jangan push file `.pt` ke GitHub. File ini sudah ada di `.gitignore`.

### 3c. Deploy di Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **New app**
4. Isi form:
   - **Repository:** `username/capstone4-safety`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Klik **Deploy**
6. Tunggu build selesai (3-10 menit)
7. Saat pertama kali dibuka, model otomatis didownload dari Google Drive

---

## Fitur Aplikasi

- **Single Image Mode:** Upload 1 gambar → deteksi + compliance report
- **Batch Mode:** Upload beberapa gambar → analisis sekaligus
- **IoU-based Person Matching:** Setiap pekerja dianalisis secara individual
- **Safety Compliance Score:** Persentase pekerja yang lengkap APD
- **Risk Level:** LOW / MEDIUM / HIGH berdasarkan compliance rate
- **Detail per Pekerja:** Status COMPLIANT atau MELANGGAR per individu
- **Download Hasil:** Simpan gambar dengan bounding box deteksi

## Classes yang Dideteksi

| ID | Class | Keterangan |
|----|-------|-----------|
| 0 | helmet | Pekerja memakai helm |
| 1 | no-helmet | Pekerja TIDAK memakai helm |
| 2 | no-vest | Pekerja TIDAK memakai rompi |
| 3 | person | Deteksi pekerja |
| 4 | vest | Pekerja memakai rompi |
