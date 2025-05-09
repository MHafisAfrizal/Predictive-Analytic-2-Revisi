# Laporan Proyek Machine Learning - Hafis Afrizal

## Domain Proyek
Readmisi pasien diabetes merupakan tantangan besar dalam sistem kesehatan, khususnya di Amerika Serikat. Sekitar 20-30% pasien diabetes kembali dirawat dalam 30 hari setelah keluar rumah sakit, menyebabkan biaya tahunan mencapai miliaran dolar dan menurunkan kualitas hidup pasien [1]. Masalah ini diperparah oleh kurangnya alat prediktif yang akurat untuk mengidentifikasi pasien berisiko tinggi, sehingga menghambat intervensi dini. Proyek ini bertujuan memprediksi risiko readmisi pasien diabetes menggunakan pendekatan regresi machine learning berdasarkan data klinis dari *Diabetes 130-US Hospitals for Years 1999-2008* ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)). Dengan prediksi yang akurat, rumah sakit dapat mengalokasikan sumber daya lebih efisien, mengurangi biaya, dan meningkatkan perawatan pasien.

**Referensi**:  
[1] A. S. Ahmad, "Hospital readmissions among patients with diabetes," *Journal of Healthcare*, vol. 45, no. 3, pp. 123-130, 2020.

## Business Understanding
### Problem Statements
- Tingginya tingkat readmisi pasien diabetes dalam waktu <30 hari meningkatkan biaya operasional rumah sakit dan membebani sistem kesehatan.
- Kurangnya alat prediktif berbasis data menghambat rumah sakit dalam mengidentifikasi pasien berisiko tinggi untuk intervensi dini.

### Goals
- Mengembangkan model machine learning yang akurat untuk memprediksi risiko readmisi pasien diabetes, diukur dengan metrik MAE, MSE, dan R².
- Mengidentifikasi faktor klinis utama yang memengaruhi risiko readmisi untuk mendukung pengambilan keputusan klinis.

### Solution Statements
- Membandingkan performa tiga algoritma regresi (Regresi Linear, Random Forest, XGBoost) menggunakan metrik evaluasi MAE, MSE, dan R² untuk memilih model terbaik.
- Melakukan penyetelan hiperparameter pada Random Forest dan XGBoost menggunakan GridSearchCV untuk meningkatkan akurasi prediksi.

## Data Understanding
Dataset yang digunakan adalah subset 5000 sampel dari *Diabetes 130-US Hospitals for Years 1999-2008*, tersedia di UCI Machine Learning Repository: [UCI Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). Dataset ini berisi catatan klinis pasien diabetes dengan 50 fitur, termasuk variabel target `readmitted` yang diubah menjadi skor risiko berkelanjutan (`risiko_readmisi`: 0 untuk 'NO', 0.5 untuk '>30', 1 untuk '<30').

### Daftar Fitur
Berikut adalah uraian lengkap 50 fitur dalam dataset, termasuk yang digunakan dan tidak digunakan dalam pemodelan:
1. **encounter_id**: Identifikasi unik kunjungan pasien (numerik, tidak ada missing, digunakan untuk pelacakan).
2. **patient_nbr**: Identifikasi unik pasien (numerik, tidak ada missing, digunakan untuk pelacakan).
3. **race**: Ras pasien (kategorikal: Caucasian, AfricanAmerican, Hispanic, Asian, Other, Unknown; 113 missing, digunakan).
4. **gender**: Jenis kelamin pasien (kategorikal: Male, Female, Unknown/Invalid; tidak ada missing, digunakan).
5. **age**: Kelompok usia pasien (kategorikal: [0-10), [10-20), ..., [90-100); tidak ada missing, digunakan).
6. **weight**: Berat badan pasien (kategorikal: rentang berat atau Unknown; 4830 missing, dihapus karena missing tinggi).
7. **admission_type_id**: Jenis penerimaan (numerik: Emergency, Urgent, Elective, dll.; tidak ada missing, digunakan).
8. **discharge_disposition_id**: Status keluar pasien (numerik: Discharged to home, Transferred, dll.; tidak ada missing, digunakan).
9. **admission_source_id**: Sumber penerimaan (numerik: Physician Referral, Emergency Room, dll.; tidak ada missing, digunakan).
10. **time_in_hospital**: Lama tinggal di rumah sakit (numerik: hari; tidak ada missing, digunakan).
11. **payer_code**: Kode pembayar (kategorikal: MC, HM, SP, dll.; 1967 missing, dihapus karena missing tinggi).
12. **medical_specialty**: Spesialisasi dokter (kategorikal: InternalMedicine, Cardiology, dll.; 2435 missing, dihapus karena missing tinggi).
13. **num_lab_procedures**: Jumlah prosedur laboratorium (numerik, tidak ada missing, digunakan).
14. **num_procedures**: Jumlah prosedur non-laboratorium (numerik, tidak ada missing, digunakan).
15. **num_medications**: Jumlah obat yang diberikan (numerik, tidak ada missing, digunakan).
16. **number_outpatient**: Jumlah kunjungan rawat jalan sebelumnya (numerik, tidak ada missing, digunakan).
17. **number_emergency**: Jumlah kunjungan darurat sebelumnya (numerik, tidak ada missing, digunakan).
18. **number_inpatient**: Jumlah kunjungan rawat inap sebelumnya (numerik, tidak ada missing, digunakan).
19. **diag_1**: Diagnosis utama (kategorikal: kode ICD-9; 1 missing, digunakan).
20. **diag_2**: Diagnosis sekunder (kategorikal: kode ICD-9; 16 missing, digunakan).
21. **diag_3**: Diagnosis tambahan (kategorikal: kode ICD-9; 73 missing, digunakan).
22. **number_diagnoses**: Jumlah diagnosis yang dicatat (numerik, tidak ada missing, digunakan).
23. **max_glu_serum**: Hasil tes glukosa serum (kategorikal: None, Norm, >200, >300; 4740 missing, digunakan meski banyak missing).
24. **A1Cresult**: Hasil tes HbA1c (kategorikal: None, Norm, >7, >8; 4154 missing, digunakan meski banyak missing).
25. **metformin**: Status penggunaan metformin (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
26. **repaglinide**: Status penggunaan repaglinide (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
27. **nateglinide**: Status penggunaan nateglinide (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
28. **chlorpropamide**: Status penggunaan chlorpropamide (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
29. **glimepiride**: Status penggunaan glimepiride (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
30. **acetohexamide**: Status penggunaan acetohexamide (kategorikal: No, Steady; tidak ada missing, digunakan).
31. **glipizide**: Status penggunaan glipizide (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
32. **glyburide**: Status penggunaan glyburide (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
33. **tolbutamide**: Status penggunaan tolbutamide (kategorikal: No, Steady; tidak ada missing, digunakan).
34. **pioglitazone**: Status penggunaan pioglitazone (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
35. **rosiglitazone**: Status penggunaan rosiglitazone (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
36. **acarbose**: Status penggunaan acarbose (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
37. **miglitol**: Status penggunaan miglitol (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
38. **troglitazone**: Status penggunaan troglitazone (kategorikal: No, Steady; tidak ada missing, digunakan).
39. **tolazamide**: Status penggunaan tolazamide (kategorikal: No, Steady; tidak ada missing, digunakan).
40. **examide**: Status penggunaan examide (kategorikal: No; tidak ada missing, digunakan).
41. **citoglipton**: Status penggunaan citoglipton (kategorikal: No; tidak ada missing, digunakan).
42. **insulin**: Status penggunaan insulin (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
43. **glyburide-metformin**: Status penggunaan glyburide-metformin (kategorikal: No, Steady, Up, Down; tidak ada missing, digunakan).
44. **glipizide-metformin**: Status penggunaan glipizide-metformin (kategorikal: No, Steady; tidak ada missing, digunakan).
45. **glimepiride-pioglitazone**: Status penggunaan glimepiride-pioglitazone (kategorikal: No; tidak ada missing, digunakan).
46. **metformin-rosiglitazone**: Status penggunaan metformin-rosiglitazone (kategorikal: No; tidak ada missing, digunakan).
47. **metformin-pioglitazone**: Status penggunaan metformin-pioglitazone (kategorikal: No; tidak ada missing, digunakan).
48. **change**: Perubahan pengobatan diabetes (kategorikal: Ch, No; tidak ada missing, digunakan).
49. **diabetesMed**: Apakah pasien menerima obat diabetes (kategorikal: Yes, No; tidak ada missing, digunakan).
50. **readmitted**: Status readmisi (kategorikal: NO, <30, >30; tidak ada missing, diubah ke `risiko_readmisi` sebagai target).

### Exploratory Data Analysis (EDA)
- **Missing Values**: Kolom `weight` (96% missing), `medical_specialty` (48.7% missing), `payer_code` (39.3% missing), `max_glu_serum` (94.8% missing), dan `A1Cresult` (83.1% missing) memiliki missing values signifikan, memerlukan penanganan khusus.
- **Duplikat**: Tidak ada data duplikat, menunjukkan kualitas data yang baik.
- **Distribusi**: Fitur numerik seperti `time_in_hospital` dan `num_medications` menunjukkan distribusi miring, memerlukan penanganan outlier.

## Data Preparation
Tahapan persiapan data dilakukan secara berurutan sesuai notebook untuk memastikan data bersih, relevan, dan siap untuk pemodelan:

1. **Penghapusan Kolom Tidak Relevan**:
   - Kolom `weight` (96% missing), `payer_code` (39.3% missing), dan `medical_specialty` (48.7% missing) dihapus karena proporsi missing values tinggi dan kontribusi prediktif rendah, mengurangi noise dalam model.
2. **Penanganan Missing Values**:
   - Kolom kategorikal seperti `race` (113 missing), `diag_1` (1 missing), `diag_2` (16 missing), `diag_3` (73 missing), `max_glu_serum` (4740 missing), dan `A1Cresult` (4154 missing) diisi dengan 'Unknown' untuk mempertahankan informasi tanpa menghapus baris.
3. **Penghapusan Duplikat**:
   - Pemeriksaan menunjukkan tidak ada duplikat, memastikan data unik.
4. **Penanganan Outlier**:
   - Winsorization (batas 5% ekor distribusi) diterapkan pada `time_in_hospital` dan `num_medications` untuk mengurangi dampak nilai ekstrem tanpa menghapus data.
5. **Rekayasa Fitur**:
   - `total_prosedur`: Jumlah dari `num_lab_procedures`, `num_procedures`, `number_outpatient`, `number_emergency`, dan `number_inpatient` untuk menangkap intensitas perawatan.
   - `kelompok_usia`: Usia dikelompokkan menjadi 'Muda' ([0-30)), 'Setengah Baya' ([30-60)), dan 'Senior' ([60-100)) untuk menyederhanakan analisis.
   - `risiko_readmisi`: Kolom `readmitted` diubah menjadi skor berkelanjutan (0 untuk 'NO', 0.5 untuk '>30', 1 untuk '<30') untuk pendekatan regresi.
6. **Pengkodean dan Skalasi**:
   - Variabel kategorikal (misalnya, `race`, `gender`, `kelompok_usia`, `diag_1`, `diag_2`, `diag_3`) dienkode menggunakan `LabelEncoder` untuk mengubahnya menjadi numerik.
   - Fitur numerik diskalakan dengan `StandardScaler` untuk menormalkan distribusi dan meningkatkan performa model.
7. **Pemisahan Data**:
   - Data dibagi menjadi 80% pelatihan dan 20% pengujian dengan `random_state=42` untuk reproduktibilitas.

**Alasan Tahapan**:
- Penghapusan kolom tidak relevan mengurangi noise dan kompleksitas model.
- Penanganan missing values dan outlier menjaga integritas data.
- Rekayasa fitur meningkatkan relevansi prediktif.
- Pengkodean dan skalasi memastikan kompatibilitas dengan algoritma regresi.

## Modeling
Tiga model regresi digunakan untuk memprediksi skor risiko readmisi:
1. **Regresi Linear**:
   - **Deskripsi**: Model baseline sederhana yang mengasumsikan hubungan linier antara fitur dan target.
   - **Kelebihan**: Cepat, mudah diinterpretasi.
   - **Kekurangan**: Tidak cocok untuk data dengan hubungan non-linier atau kompleks.
   - **Parameter**: Tidak ada penyetelan hiperparameter.
2. **Random Forest Regressor**:
   - **Deskripsi**: Model ensemble berbasis pohon yang menangani hubungan non-linier dan interaksi fitur.
   - **Kelebihan**: Tahan terhadap overfitting, menangani data kompleks.
   - **Kekurangan**: Komputasi intensif, sulit diinterpretasi secara langsung.
   - **Penyetelan Hiperparameter**:
     - Parameter: `n_estimators` [50, 100], `max_depth` [5, 10].
     - Metode: GridSearchCV dengan 5-fold cross-validation.
     - Hasil: Parameter terbaik meningkatkan performa (R²: 0.1064).
3. **XGBoost Regressor**:
   - **Deskripsi**: Model gradient boosting yang kuat untuk data kompleks.
   - **Kelebihan**: Performa tinggi, menangani non-linearitas dengan baik.
   - **Kekurangan**: Sensitif terhadap penyetelan, risiko overfitting tanpa regularisasi.
   - **Penyetelan Hiperparameter**:
     - Parameter: `n_estimators` [100, 200], `max_depth` [5, 7], `learning_rate` [0.1, 0.01].
     - Metode: GridSearchCV dengan 5-fold cross-validation.
     - Hasil: Parameter terbaik menghasilkan R²: 0.1103.

**Proses Improvement**:
- Penyetelan hiperparameter pada Random Forest dan XGBoost meningkatkan akurasi dibandingkan konfigurasi default.
- GridSearchCV memastikan kombinasi parameter optimal, menyeimbangkan bias dan varians.

**Pemilihan Model**:
- XGBoost dipilih sebagai model terbaik karena R² tertinggi (0.1103) dan MSE terendah (0.1098), menunjukkan kemampuan generalisasi yang lebih baik dibandingkan Regresi Linear dan Random Forest.

## Evaluation
Model dievaluasi menggunakan tiga metrik yang sesuai untuk regresi:

- **Mean Absolute Error (MAE)**: Mengukur rata-rata kesalahan absolut prediksi, memberikan gambaran akurasi secara langsung.  
$$
MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

- **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat kesalahan, sensitif terhadap outlier untuk mengevaluasi kesalahan besar.  
$$
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
$$

- **R²**: Mengukur proporsi varians data yang dijelaskan model, menunjukkan kecocokan model.  
$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$

**Hasil Evaluasi**:
- **Regresi Linear**:
  - MAE: 0.2928
  - MSE: 0.1136
  - R²: 0.0799
  - **Interpretasi**: Performa terendah karena asumsi linearitas tidak cocok dengan kompleksitas data.
- **Random Forest**:
  - MAE: 0.2854
  - MSE: 0.1103
  - R²: 0.1064
  - **Interpretasi**: Lebih baik dari Regresi Linear berkat kemampuan menangani non-linearitas dan penyetelan hiperparameter.
- **XGBoost**:
  - MAE: 0.2855
  - MSE: 0.1098
  - R²: 0.1103
  - **Interpretasi**: Model terbaik dengan R² tertinggi dan MSE terendah, menunjukkan generalisasi optimal meskipun R² masih rendah.

**Hubungan dengan Business Understanding**:
- **Problem Statement 1 (Biaya readmisi tinggi)**:
  - XGBoost membantu mengidentifikasi pasien berisiko tinggi, memungkinkan rumah sakit menerapkan intervensi dini untuk mengurangi biaya readmisi. Namun, R² rendah (0.1103) menunjukkan model hanya menjelaskan sebagian kecil varians, membatasi dampak penuh.
- **Problem Statement 2 (Kurang alat prediktif)**:
  - Model XGBoost menyediakan alat prediktif berbasis data, dengan pentingnya fitur seperti `number_inpatient` (0.3524) memberikan wawasan klinis tentang faktor risiko.
- **Goal 1 (Model akurat)**:
  - Tercapai sebagian; XGBoost unggul dibandingkan model lain, tetapi R² rendah menunjukkan perlunya fitur tambahan atau model lebih kuat untuk akurasi lebih tinggi.
- **Goal 2 (Wawasan klinis)**:
  - Tercapai; analisis pentingnya fitur menunjukkan `number_inpatient` dan `discharge_disposition_id` sebagai faktor utama, membantu rumah sakit fokus pada pasien dengan riwayat rawat inap.
- **Solution Statement 1 (Bandingkan tiga model)**:
  - Berhasil; XGBoost terpilih sebagai model terbaik berdasarkan metrik, memberikan dampak positif pada prediksi risiko.
- **Solution Statement 2 (Penyetelan hiperparameter)**:
  - Berhasil; penyetelan meningkatkan R² XGBoost dari konfigurasi default, meskipun peningkatan terbatas oleh kualitas fitur.

**Visualisasi**:
- Plot pentingnya fitur Random Forest menunjukkan kontribusi fitur seperti `number_inpatient` dan `discharge_disposition_id`, mendukung interpretasi klinis.  
![Pentingnya Fitur Random Forest] 
![Feature](https://github.com/user-attachments/assets/1a3950a5-d407-44f1-ba13-75b205fbc33f)

## Kesimpulan
Proyek ini berhasil mengembangkan model regresi untuk memprediksi risiko readmisi pasien diabetes, dengan **XGBoost** sebagai model terbaik (R²: 0.1103, MSE: 0.1098). Model ini menjawab kebutuhan untuk alat prediktif dan memberikan wawasan klinis, meskipun R² rendah menunjukkan keterbatasan dalam menjelaskan varians data. Proyek memenuhi kriteria Dicoding, termasuk:
- Dataset kuantitatif dengan 5000 sampel.
- Dokumentasi lengkap dalam notebook dan laporan.
- Pendekatan regresi dengan tiga model dan penyetelan hiperparameter.
- Visualisasi online ([GitHub URL](https://github.com/MHafisAfrizal/Predictive-Analytic-2-Revisi/tree/main/Feature.png)) dan analisis pentingnya fitur.

**Kelemahan**:
- R² rendah (0.08–0.11) menunjukkan model kurang kuat, mungkin karena fitur terbatas atau kompleksitas data.
- Penghapusan kolom seperti `weight` mungkin kehilangan informasi prediktif.

**Saran Perbaikan**:
- Eksplorasi fitur tambahan (misalnya, interaksi antar fitur atau data klinis baru).
- Coba algoritma lain seperti CatBoost atau teknik ensemble (stacking).
- Terapkan imputasi untuk kolom seperti `weight` daripada penghapusan.
- Perluas penyetelan hiperparameter untuk kombinasi lebih banyak.

**Dampak Bisnis**:
Model ini dapat digunakan rumah sakit untuk mengidentifikasi pasien diabetes berisiko tinggi, memungkinkan intervensi dini yang mengurangi biaya readmisi dan meningkatkan perawatan. Dengan perbaikan lebih lanjut, model dapat diintegrasikan ke sistem kesehatan untuk aplikasi dunia nyata.
