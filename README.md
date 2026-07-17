# 🍅 UAS KECERDASAN BUATAN
# Deteksi Penyakit Daun Tomat Menggunakan Convolutional Neural Network (CNN) dan MobileNetV2

---

## 📖 Deskripsi Proyek

Proyek ini merupakan tugas akhir mata kuliah **Kecerdasan Buatan** yang bertujuan untuk membangun sistem klasifikasi penyakit daun tomat menggunakan metode **Deep Learning**. Penelitian ini membandingkan performa dua model, yaitu **Convolutional Neural Network (CNN)** dan **MobileNetV2**, dalam mengklasifikasikan citra daun tomat menjadi tiga kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**.

---

# 📑 Daftar Isi

- Latar Belakang
- Tujuan Proyek
- Dataset
- Struktur Repository
- Teknologi yang Digunakan
- Metodologi
- Model Deep Learning
- Hasil Pengujian
- Perbandingan Model
- Cara Menjalankan Project
- Referensi
- Penulis

---

# 🌱 Latar Belakang

Penyakit pada daun tomat menjadi salah satu penyebab utama menurunnya hasil panen. Identifikasi penyakit secara manual membutuhkan pengalaman dan waktu yang cukup lama sehingga sering terjadi keterlambatan penanganan. Oleh karena itu, penelitian ini memanfaatkan teknologi **Deep Learning** untuk membantu proses identifikasi penyakit daun tomat secara otomatis melalui citra digital.

---

# 🎯 Tujuan Proyek

- Mengembangkan sistem klasifikasi penyakit daun tomat berbasis Deep Learning.
- Membandingkan performa model CNN dan MobileNetV2.
- Menentukan model terbaik berdasarkan hasil evaluasi.
- Membantu proses identifikasi penyakit daun tomat secara lebih cepat dan akurat.

---

# 📂 Dataset

Dataset yang digunakan pada penelitian ini adalah **Tomato Disease Multiple Sources** yang tersedia di Kaggle.

Dataset ini berisi lebih dari **20.000 citra daun tomat** yang terdiri atas **10 kelas penyakit** dan **1 kelas daun sehat (Healthy)**. Dataset merupakan gabungan dari beberapa sumber, termasuk PlantVillage dan citra daun tomat yang diambil di lingkungan nyata (in-the-wild), serta telah melalui proses augmentasi sehingga memiliki variasi data yang lebih beragam. :contentReference[oaicite:0]{index=0}

Pada penelitian ini hanya digunakan **3 kelas**, yaitu:

- 🌿 Healthy
- 🍂 Early Blight
- 🍁 Late Blight

Struktur dataset yang digunakan:

```text
dataset/
│
├── train/
│   ├── Healthy/
│   ├── Early_blight/
│   └── Late_blight/
│
└── valid/
    ├── Healthy/
    ├── Early_blight/
    └── Late_blight/
```

**Sumber Dataset:**

https://www.kaggle.com/datasets/arjuntejaswi/plant-village

```

---

# 📁 Struktur Repository

```text
UAS-KecerdasanBuatan/
│
├── Data/
│   ├── Dataset/
│   └── Jurnal/
│
├── Laporan_uas.md
├── README.md
├── UAS_KecerdasanBuatan_model_CNN.ipynb
└── UAS_KecerdasanBuatan_model_MobileNetV2.ipynb
```

---

# 💻 Teknologi yang Digunakan

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Google Colab

---

# 🔬 Metodologi

Tahapan penelitian yang dilakukan yaitu:

1. Pengumpulan Dataset
2. Exploratory Data Analysis (EDA)
3. Data Preparation
4. Data Augmentation
5. Pembuatan Model CNN
6. Pembuatan Model MobileNetV2
7. Training Model
8. Evaluasi Model
9. Perbandingan Hasil
10. Kesimpulan

---

# 🧠 Model Deep Learning

## Model 1

**Convolutional Neural Network (CNN)**

Digunakan sebagai model utama yang dibangun menggunakan beberapa layer Conv2D, MaxPooling2D, Flatten, Dense, dan Dropout untuk melakukan ekstraksi fitur citra secara otomatis.

---

## Model 2

**MobileNetV2**

Menggunakan metode **Transfer Learning** dengan bobot awal ImageNet sehingga proses pelatihan menjadi lebih efisien dan memiliki performa yang baik pada klasifikasi citra.

---

# 📊 Hasil Pengujian

Model dievaluasi menggunakan beberapa metrik:

- Accuracy
- Loss
- Precision
- Recall
- F1-Score
- Confusion Matrix

> **Hasil evaluasi akan ditambahkan setelah proses training selesai.**

---

# 📈 Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| CNN | - | - | - | - |
| MobileNetV2 | - | - | - | - |

---

# 🚀 Cara Menjalankan Project

1. Clone repository

```bash
git clone https://github.com/sapaat06/UAS-KecerdasanBuatan.git
```

2. Install library

```bash
pip install tensorflow matplotlib numpy pandas scikit-learn opencv-python
```

3. Jalankan notebook

- `UAS_KecerdasanBuatan_model_CNN.ipynb`
- `UAS_KecerdasanBuatan_model_MobileNetV2.ipynb`

---

# 📚 Referensi

- Howard et al. (2017). MobileNets.
- Sandler et al. (2018). MobileNetV2.
- Mohanty et al. (2016). Using Deep Learning for Image-Based Plant Disease Detection.
- TensorFlow Documentation.
- PlantVillage Dataset.

---

# 👨‍💻 Penulis

**Nama:** Sapaat

**Mata Kuliah:** Kecerdasan Buatan

**Program Studi:** Teknik Informatika

**Institut Teknologi Garut**

**Tahun:** 2026

---

## ⭐ Hasil Akhir

Setelah training selesai, README ini akan diperbarui dengan:

- 📷 Contoh dataset
- 📊 Grafik Accuracy & Loss CNN
- 📊 Grafik Accuracy & Loss MobileNetV2
- 📈 Grafik perbandingan kedua model
- 🔥 Confusion Matrix CNN
- 🔥 Confusion Matrix MobileNetV2
- 📋 Classification Report
- 🏆 Model terbaik berdasarkan hasil evaluasi
