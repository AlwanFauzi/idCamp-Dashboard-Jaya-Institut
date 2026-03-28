
# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Institut

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

**Sumber Data:**

Dataset utama diperoleh dari repositori Dicoding Academy:

- [**Dataset Utama (Dicoding)**](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv) - Data performa dan status mahasiswa

Sumber data ini menyediakan informasi komprehensif terkait karakteristik demografis, finansial, dan akademik mahasiswa untuk melatih model prediksi dropout.

Setup environment:
```powershell
python -m venv venv
venv\Scripts\activate
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

Langkah-langkah akses dashboard:
1. Buka link Looker Studio di browser.
2. Tunggu sampai seluruh komponen visual selesai dimuat.
3. Gunakan filter/periode yang tersedia untuk mengeksplorasi indikator dropout.
4. Jika muncul permintaan akses, login menggunakan akun Google lalu refresh halaman.

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

Ringkasan metrik model (berdasarkan model/model_metrics.json) - dilatih hanya pada data Dropout dan Graduate:

| Metrik | Nilai |
|--------|-------|
| **Threshold** | 0.45 |
| **ROC-AUC** | 0.975 |
| **Accuracy** | 0.908 |
| **Precision** | 0.840 |
| **Recall** | 0.944 |
| **F1-Score** | 0.889 |

**Penjelasan Metrik:**
- **ROC-AUC (0.975)**: Model menunjukkan kemampuan pembeda yang sangat baik antara mahasiswa dengan risiko dropout dan tidak dropout di berbagai threshold.
- **Recall (0.944)**: Dari seluruh mahasiswa yang benar-benar dropout, model mampu mengidentifikasi 94.4% dari mereka. Ini penting untuk early warning system agar tidak melewatkan kasus dropout.
- **Precision (0.840)**: Dari mahasiswa yang diprediksi berisiko dropout, 84% di antaranya memang benar-benar memiliki risiko dropout.
- **Accuracy (0.908)**: Keseluruhan akurasi prediksi untuk kedua kelas mencapai 90.8%.

Link prototype deployment:
https://idcamp-dashboard-jaya-institut.streamlit.app/

Langkah-langkah akses prototype machine learning:
1. Buka link deployment Streamlit di browser.
2. Pada halaman utama, isi form prediksi single atau gunakan menu upload CSV untuk batch prediction.
3. Untuk batch prediction, gunakan struktur kolom yang sama dengan data/sample_input.csv.
4. Lihat hasil prediksi pada output probabilitas, label, dan level risiko.
5. Jika aplikasi sedang sleep, tunggu beberapa detik hingga proses warm-up selesai lalu refresh.

## Conclusion

### Ringkasan Temuan Analisis

Proyek ini berhasil mengidentifikasi dan memodelkan faktor-faktor utama yang mempengaruhi risiko dropout mahasiswa di Jaya Jaya Institut. Berikut temuan kunci berdasarkan analisis data:

#### 1. Faktor Akademik (Paling Berpengaruh)
- **MK Lulus Semester 2** adalah faktor paling penting dalam prediksi dropout, diikuti oleh **MK Lulus Semester 1**.
- Mahasiswa yang lulus lebih banyak mata kuliah cenderung memiliki risiko dropout yang jauh lebih rendah.
- analisis menunjukkan adanya perbedaan signifikan dalam performa akademik antara mahasiswa yang lulus dan yang dropout.

#### 2. Faktor Finansial
- **Pembayaran biaya kuliah yang tertunggak** menunjukkan korelasi kuat dengan risiko dropout (dropout rate mencapai **88% untuk mahasiswa yang biaya tidak up-to-date**).
- **Status debtor** juga berpengaruh signifikan terhadap risiko dropout.
- Mahasiswa tanpa beasiswa menunjukkan dropout rate yang lebih tinggi.

#### 3. Faktor Demografi  
- **Usia masuk mahasiswa** mempengaruhi risiko dropout:
  - Mahasiswa kelompok 26-30 tahun memiliki dropout rate tertinggi **70.2%**
  - Mahasiswa kelompok ≤18 tahun memiliki dropout rate terendah **23.3%**
  - Risiko dropout meningkat seiring bertambahnya usia masuk.

### Performa Model

Model **Logistic Regression** yang dibangun menunjukkan hasil yang sangat memuaskan:
- **ROC-AUC 0.975**: Kemampuan diskriminasi yang sangat baik
- **Recall 0.944**: Mampu mendeteksi 94.4% mahasiswa dropout (tingkat deteksi sangat tinggi)
- **Precision 0.840**: 84% dari prediksi "akan dropout" adalah benar

Dengan performa ini, model dapat digunakan sebagai **alat deteksi dini yang andal** untuk mengidentifikasi mahasiswa berisiko dropout.

### Keterbatasan Sistem

1. **Data Historis**: Model dilatih menggunakan data mahasiswa yang sudah lulus atau dropout. Aplikasi pada mahasiswa Enrolled memerlukan validasi tambahan.
2. **Periode Pengamatan**: Data mencakup periode akademik tertentu. Perubahan kebijakan atau kondisi institusi mungkin mempengaruhi akurasi model.
3. **Dashboard**: Representasi visual dashboard menampilkan snapshot data historis dan perlu pembaruan berkala sesuai data terbaru.
4. **Model Dependen pada Fitur**: Akurasi model bergantung pada ketersediaan data fitur-fitur akademik dan finansial yang lengkap.

### Arah Pengembangan Ke Depan

1. **Monitoring Berkala**: Evaluasi performa model secara berkala dengan data baru untuk memastikan akurasi tetap tinggi.
2. **Integrasi Sistem**: Integrasikan model dengan sistem akademik institusi untuk monitoring real-time dan alert otomatis.
3. **Strategi Intervensi**: Kembangkan strategi intervensi berbasis segmentasi risiko yang telah diidentifikasi.
4. **Evaluasi Dampak**: Ukur dampak dari program intervensi terhadap pengurangan tingkat dropout.
5. **Eksplorasi Fitur Tambahan**: Pertimbangkan fitur-fitur kualitatif (misal: survey kepuasan, mentoring history) yang mungkin meningkatkan predictive power.

## Rekomendasi Action Items
Berdasarkan hasil analisis dan model yang telah dibangun, berikut rekomendasi action items spesifik yang berbasis data:

### 1. **Intervensi Finansial - Prioritas Utama**
**Temuan:** Mahasiswa dengan biaya kuliah tertunggak (not up-to-date) menunjukkan **dropout rate 88%**, dan status debtor memiliki dropout rate 78%.

**Action Items:**
- **Program Beasiswa Darurat**: Identifikasi mahasiswa dengan tunggakan biaya >3 bulan, berikan akses ke jalur beasiswa darurat atau skema cicilan fleksibel.
- **Monitoring Pembayaran**: Lakukan monitoring rutin pembayaran biaya kuliah setiap akhir semester untuk deteksi dini mahasiswa bermasalah.
- **Kerjasama dengan Layanan Keuangan**: Fasilitasi akses mahasiswa untuk mendapatkan bantuan finansial dari pihak ketiga jika diperlukan.

### 2. **Intervensi Akademik - Prioritas Tinggi**
**Temuan:** Performa akademik semester awal (MK lulus, nilai) adalah faktor paling berpengaruh. Mahasiswa dengan <3 MK lulus di semester 1 menunjukkan sinyal risiko dropout yang kuat.

**Action Items:**
- **Tutorial & Mentoring Semester Dini**: Mulai program tutorial kelompok kecil dan mentoring 1-on-1 sejak minggu pertama berdasarkan hasil placement test.
- **Early Alert System**: Gunakan model untuk mengidentifikasi mahasiswa dengan risiko tinggi saat akhir semester 1, kemudian berikan intervensi intensif di semester 2.
- **Review Kurikulum & Metode Pengajaran**: Evaluasi mata kuliah dengan tingkat kegagalan tinggi untuk perbaikan metode pengajaran atau struktur materi.

### 3. **Targeting Demografis - Prioritas Sedang**
**Temuan:** Mahasiswa kelompok usia 26-30 tahun memiliki dropout rate tertinggi (70.2%), diikuti kelompok >30 tahun (61.4%).

**Action Items:**
- **Program Khusus Mahasiswa Berusia Lanjut**: Sediakan program dukungan khusus bagi mahasiswa berusia >25 tahun, termasuk fleksibilitas jadwal dan dukungan khusus.
- **Community Building**: Fasilitasi pembentukan peer support groups untuk mahasiswa yang lebih matang secara usia.

### 4. **Implementasi Early Warning System - Prioritas Tinggi**
**Temuan:** Model yang dibangun memiliki recall 94.4%, mampu mendeteksi sebagian besar mahasiswa berisiko dropout.

**Action Items:**
- **Deployment Model**: Deploy model pada platform akademik atau dashboard untuk monitoring otomatis.
- **Alert Rutin**: Setup alert otomatis setiap akhir semester untuk mendeteksi mahasiswa dengan probabilitas dropout tinggi (>65%).
- **Triage Intervensi**: Buat protokol respons cepat untuk mahasiswa alert, termasuk panggilan akademik dan penawaran program dukungan.

### 5. **Validasi & Perbaikan Berkelanjutan - Prioritas Berkelanjutan**
**Temuan:** Model ini adalah prototipe yang perlu divalidasi terus-menerus untuk memastikan relevansi dan akurasi.

**Action Items:**
- **Validasi Manual**: Untuk 10-20% kasus prediksi, lakukan validasi tambahan dengan melakukan wawancara atau assessment kualitatif.
- **Feedback Loop**: Kumpulkan feedback dari akademik tentang mana mahasiswa yang benar-benar dropout dan gunakan untuk perbaikan model.
- **Re-training Berkala**: Lakukan retraining model setiap semester atau tahunan dengan data terbaru untuk menjaga akurasi.