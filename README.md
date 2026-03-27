
# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut menghadapi tantangan tingginya mahasiswa yang tidak menyelesaikan studi (dropout). Kondisi ini berdampak pada kualitas performa akademik institusi, efektivitas proses pembinaan mahasiswa, dan pengambilan keputusan manajerial. Untuk itu, dibutuhkan solusi berbasis data agar risiko dropout dapat dideteksi lebih dini.

Dalam proyek ini, solusi dibangun dalam dua bentuk:
1. Business dashboard untuk memantau indikator utama terkait dropout.
2. Prototype sistem machine learning untuk memprediksi risiko dropout mahasiswa.

### Permasalahan Bisnis
Permasalahan bisnis yang diselesaikan pada proyek ini:
1. Sulitnya mengidentifikasi mahasiswa berisiko dropout secara dini.
2. Belum adanya ringkasan visual (dashboard) yang memudahkan monitoring faktor-faktor penting dropout.
3. Belum tersedianya prototipe prediksi yang mudah digunakan oleh pengguna non-teknis.

### Cakupan Proyek
Cakupan pekerjaan pada proyek ini meliputi:
1. Analisis data mahasiswa untuk memahami pola dropout.
2. Menyiapkan data untuk pemodelan klasifikasi biner (dropout vs non-dropout).
3. Membangun dan menyimpan pipeline model machine learning.
4. Menyusun business dashboard untuk kebutuhan monitoring.
5. Membangun aplikasi Streamlit untuk prediksi single dan batch.

### Persiapan
Sumber data:
1. Dataset utama: data/data.csv
2. Template input batch: data/sample_input.csv
3. Dataset dashboard: dashboard/dashboard_dataset_metabase_clean.csv

Setup environment:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Struktur proyek saat ini:
```text
idCamp-Dashboard-Jaya-Institut/
├── .streamlit/
│   └── config.toml
├── dashboard/
│   ├── alwanfauzi-dashboard.png
│   ├── dashboard_dataset_metabase_clean.csv
│   └── url.txt
├── data/
│   ├── data.csv
│   └── sample_input.csv
├── model/
│   ├── dropout_pipeline.joblib
│   ├── feature_metadata.json
│   └── model_metrics.json
├── app.py
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Business Dashboard
Business dashboard dibuat untuk membantu pemantauan cepat indikator utama terkait mahasiswa dan dropout, antara lain distribusi status mahasiswa, faktor finansial, serta pola performa akademik awal.

Artefak dashboard:
1. Screenshot dashboard: dashboard/alwanfauzi-dashboard.png
2. Dataset dashboard: dashboard/dashboard_dataset_metabase_clean.csv
3. Link dashboard publik (tersimpan di dashboard/url.txt):

https://lookerstudio.google.com/reporting/2c30915c-324c-4b3f-91fe-cc1b2d0fb522

## Menjalankan Sistem Machine Learning
Prototype sistem machine learning tersedia pada file app.py.

Cara menjalankan prototype secara lokal:
```powershell
streamlit run app.py
```

Fitur utama prototype:
1. Prediksi cepat untuk 1 data mahasiswa.
2. Prediksi batch melalui upload file CSV.
3. Validasi kolom input batch.
4. Output probabilitas dropout, label prediksi, dan level risiko.

File model yang digunakan:
1. model/dropout_pipeline.joblib
2. model/feature_metadata.json
3. model/model_metrics.json

Ringkasan metrik model saat ini (berdasarkan model/model_metrics.json):
1. Threshold: 0.45
2. ROC-AUC: 0.931
3. Accuracy: 0.864
4. Precision: 0.755
5. Recall: 0.856
6. F1-Score: 0.802

Link prototype deployment:
Belum tersedia. Prototype saat ini dijalankan secara lokal menggunakan Streamlit.

## Conclusion
Proyek ini berhasil menyediakan solusi end-to-end untuk membantu deteksi dini risiko dropout mahasiswa di Jaya Jaya Institut. Dari sisi analitik, faktor finansial dan performa akademik semester awal menjadi indikator yang kuat terhadap kemungkinan dropout. Dari sisi implementasi, dashboard dan aplikasi prediksi sudah dapat digunakan sebagai alat bantu pengambilan keputusan awal.

### Rekomendasi Action Items
1. Susun daftar prioritas intervensi rutin untuk mahasiswa dengan probabilitas dropout tinggi.
2. Fokuskan pendampingan pada mahasiswa dengan kendala finansial dan performa akademik awal yang rendah.
3. Gunakan hasil model sebagai alat bantu keputusan awal, lalu validasi dengan evaluasi akademik/manual.