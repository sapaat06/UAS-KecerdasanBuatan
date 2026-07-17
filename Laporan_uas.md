
# LAPORAN UJIAN AKHIR SEMESTER (UAS) KECERDASAN BUATAN MENGGUNAKAN AlGORITMA DEEP LEARNING

**Mata Kuliah:** Kecerdasan Buatan


**Topik Proyek:** Deteksi Penyakit Daun Tomat Menggunakan CNN dan MobileNetV2

**Anggota Kelompok:**

1. Sapaat 2406080

   
**LINK dataset** : https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources?utm_source
---

## 1. Judul Proyek

# Analisis Perbandingan Performa Arsitektur Convolutional Neural Network (CNN) dan MobileNetV2 dalam Klasifikasi Multi-Kelas Penyakit Daun Tomat (Solanum lycopersicum)

---

## 2. Business Understanding

### Latar Belakang Masalah (Domain Proyek)

Tanaman tomat (*Solanum lycopersicum*) merupakan salah satu komoditas hortikultura yang memiliki nilai ekonomi tinggi dan banyak dibudidayakan di Indonesia. Produktivitas tanaman tomat sangat dipengaruhi oleh kondisi kesehatan tanaman, terutama pada bagian daun yang berperan penting dalam proses fotosintesis. Salah satu faktor yang menyebabkan penurunan hasil panen adalah serangan penyakit daun, seperti **Early Blight** dan **Late Blight**. Kedua penyakit tersebut dapat menyebabkan kerusakan daun, menghambat pertumbuhan tanaman, serta menurunkan kualitas dan kuantitas hasil panen apabila tidak segera ditangani.
Proses identifikasi penyakit daun tomat umumnya masih dilakukan secara manual melalui pengamatan visual oleh petani atau tenaga ahli. Metode ini membutuhkan pengalaman, waktu, dan ketelitian karena gejala antar penyakit sering kali memiliki kemiripan. Selain itu, tidak semua petani memiliki akses terhadap pakar pertanian sehingga proses diagnosis penyakit menjadi kurang efektif. Kondisi tersebut mendorong perlunya sistem yang mampu membantu proses identifikasi penyakit secara otomatis, cepat, dan akurat (Howard et al., 2017).
Perkembangan teknologi **Artificial Intelligence (AI)**, khususnya **Deep Learning**, memberikan solusi dalam bidang klasifikasi citra. Salah satu metode yang banyak digunakan adalah **Convolutional Neural Network (CNN)** karena mampu mengekstraksi fitur citra secara otomatis dan menghasilkan performa klasifikasi yang baik. Selain itu, metode **Transfer Learning** menggunakan **MobileNetV2** juga menjadi pilihan karena memiliki jumlah parameter yang lebih ringan, waktu pelatihan yang lebih cepat, serta tetap mampu memberikan akurasi yang tinggi.
Berdasarkan permasalahan tersebut, pada proyek ini dilakukan pembangunan model klasifikasi penyakit daun tomat menggunakan **CNN** dan **MobileNetV2**. Kedua model akan dibandingkan berdasarkan nilai **accuracy**, **loss**, **confusion matrix**, dan **classification report** untuk mengetahui model yang memiliki performa terbaik dalam mengklasifikasikan tiga kategori daun tomat, yaitu **Healthy**, **Early Blight**, dan **Late Blight** (Sandler et al., 2018).

### Permasalahan Dunia Nyata

1. **Kesalahan Diagnosis:** Petani sering mengalami kesulitan membedakan gejala awal antara penyakit **Early Blight** dan **Late Blight** karena keduanya memiliki karakteristik bercak yang hampir serupa. Kesalahan dalam mengidentifikasi penyakit dapat menyebabkan pemilihan jenis fungisida yang tidak tepat sehingga pengendalian penyakit menjadi kurang efektif dan berpotensi meningkatkan kerugian hasil panen.
2. **Keterbatasan Akses Pakar:** Proses identifikasi penyakit masih bergantung pada pengamatan visual oleh petani atau tenaga ahli pertanian. Di beberapa daerah, akses terhadap pakar pertanian masih terbatas sehingga proses diagnosis membutuhkan waktu lebih lama. Akibatnya, penyakit dapat menyebar ke tanaman lain sebelum dilakukan penanganan yang tepat.

### Tujuan Proyek

1. **Mengembangkan Model Klasifikasi Otomatis:** Membangun dan mengimplementasikan model berbasis Deep Learning menggunakan arsitektur **Convolutional Neural Network (CNN)** dan **MobileNetV2** untuk mendeteksi serta mengklasifikasikan penyakit pada daun tomat ke dalam tiga kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**.
2. **Membandingkan Performa Dua Model Deep Learning:** Melakukan perbandingan performa model **CNN** dan **MobileNetV2** berdasarkan metrik evaluasi seperti **Accuracy**, **Loss**, **Confusion Matrix**, dan **Classification Report** untuk mengetahui model yang memiliki kinerja terbaik.
3. **Mempelajari Karakteristik Visual Daun Tomat:** Memanfaatkan kemampuan Deep Learning dalam mengekstraksi fitur citra daun secara otomatis sehingga model mampu membedakan karakteristik visual antara daun tomat yang sehat dengan daun yang terinfeksi penyakit.
4. **Mencapai Performa Klasifikasi yang Optimal:** Menghasilkan model klasifikasi dengan tingkat akurasi yang tinggi sehingga dapat digunakan sebagai dasar pengembangan sistem deteksi penyakit daun tomat yang cepat, akurat, dan efisien.

### Target Pengguna (User)

1. **Petani Tomat:** Sebagai pengguna utama yang membutuhkan sistem untuk mendeteksi penyakit daun tomat secara cepat dan akurat melalui citra daun menggunakan perangkat komputer maupun ponsel pintar.
2. **Penyuluh Pertanian:** Sebagai alat bantu dalam melakukan identifikasi penyakit tanaman tomat di lapangan sehingga dapat memberikan rekomendasi penanganan yang lebih tepat kepada petani.
3. **Peneliti dan Mahasiswa:** Sebagai referensi dalam penelitian maupun pengembangan sistem klasifikasi penyakit tanaman berbasis Deep Learning, khususnya menggunakan metode CNN dan MobileNetV2.

### Solusi dan Manfaat

Proyek ini membangun model klasifikasi citra daun tomat berbasis **Deep Learning** menggunakan arsitektur **Convolutional Neural Network (CNN)** dan **MobileNetV2**. Manfaat yang diharapkan dari proyek ini meliputi:
- **Deteksi Penyakit Secara Cepat:** Proses identifikasi penyakit daun tomat dapat dilakukan dalam hitungan detik berdasarkan citra daun yang diinputkan.
- **Meningkatkan Akurasi Diagnosis:** Penggunaan model Deep Learning membantu mengurangi kesalahan identifikasi penyakit dibandingkan dengan metode pengamatan manual.
- **Mendukung Penanganan yang Tepat:** Hasil klasifikasi dapat membantu petani menentukan tindakan pengendalian penyakit sesuai dengan jenis penyakit yang terdeteksi sehingga dapat meminimalkan penyebaran penyakit.
- **Meningkatkan Produktivitas Pertanian:** Deteksi dini terhadap penyakit daun tomat diharapkan mampu mengurangi kerugian akibat gagal panen serta meningkatkan kualitas dan hasil produksi tomat.

---

# 3. Data Understanding

## Sumber Dataset

Dataset yang digunakan dalam proyek ini adalah **PlantVillage Dataset** yang dipublikasikan secara terbuka melalui platform Kaggle oleh kontributor **Arjun Tejaswi**. Dataset ini berisi citra daun tanaman tomat dengan berbagai kondisi kesehatan yang telah diberi label sesuai jenis penyakitnya. Pada penelitian ini hanya digunakan tiga kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**. Dataset dapat diakses melalui tautan berikut:

https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources?utm_source

## Karakteristik dan Spesifikasi Data

- **Format Berkas Asli:** Kombinasi berkas gambar berformat *.jpg* dan *.png*.
- **Dimensi Citra:** Bervariasi dengan resolusi yang berbeda pada setiap gambar.
- **Warna:** Citra berwarna 3-channel (RGB - Red, Green, Blue).
- **Target Klasifikasi:** Multi-kelas, di mana setiap gambar dikelompokkan ke dalam folder sesuai label penyakit, yaitu **Healthy**, **Early Blight**, dan **Late Blight**.
- **Struktur Dataset:** Dataset telah dipisahkan ke dalam folder **train** sebagai data pelatihan dan **valid** sebagai data validasi.
- **Transformasi Standar:** Seluruh gambar diubah ukurannya menjadi **128 × 128 × 3 piksel** agar sesuai dengan ukuran input yang digunakan pada model **CNN** dan **MobileNetV2** selama proses pelatihan.

--

## 4. Exploratory Data Analysis (EDA)

## Exploratory Data Analysis (EDA)

Proses **Exploratory Data Analysis (EDA)** dilakukan untuk memahami karakteristik dataset sebelum memasuki tahap pemodelan. Tahapan EDA pada penelitian ini meliputi:

### 1. Analisis Distribusi Kelas

Analisis dilakukan dengan menghitung jumlah citra pada masing-masing kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**. Tujuan dari analisis ini adalah untuk mengetahui distribusi jumlah data pada setiap kelas sehingga dapat diketahui apakah dataset memiliki distribusi yang seimbang. Hasil perhitungan kemudian divisualisasikan menggunakan **diagram lingkaran (pie chart)** untuk mempermudah interpretasi distribusi data.

### 2. Visualisasi Sampel Citra

Visualisasi dilakukan dengan menampilkan beberapa contoh citra dari setiap kelas. Tahap ini bertujuan untuk memahami karakteristik visual masing-masing kelas, seperti warna daun, bentuk bercak, dan pola kerusakan akibat penyakit. Dengan visualisasi ini dapat diketahui perbedaan karakteristik antara daun **Healthy**, **Early Blight**, dan **Late Blight**.

### 3. Analisis Struktur Dataset

Analisis dilakukan untuk memastikan bahwa dataset telah tersusun dengan baik sesuai kebutuhan proses pelatihan model. Dataset dibagi menjadi dua folder utama, yaitu **train** sebagai data pelatihan dan **valid** sebagai data validasi. Masing-masing folder terdiri atas tiga subfolder yang merepresentasikan kelas **Healthy**, **Early Blight**, dan **Late Blight**, sehingga dapat langsung digunakan menggunakan metode `flow_from_directory()` pada TensorFlow/Keras.

<img width="704" height="419" alt="output" src="https://github.com/user-attachments/assets/b743110a-85ff-4933-b711-472d37e05d9c" />


Grafik batang di atas menampilkan distribusi jumlah gambar setelah melewati tahap pembersihan data awal. Dataset menunjukkan kondisi yang sangat ideal (perfectly balanced dataset) di mana masing-masing dari empat kelas—yaitu Bacterial Blight, Blast, Brown Spot, dan Healthy—memiliki jumlah sampel yang sama persis, yaitu sebanyak 1.483 gambar per kelas. Keseimbangan data ini sangat menguntungkan proses pelatihan model Deep Learning karena menghilangkan risiko bias terhadap kelas tertentu, sehingga metrik akurasi yang dihasilkan nantinya akan bersifat objektif dan tepercaya tanpa perlu perlakuan khusus seperti oversampling atau undersampling



<img width="491" height="374" alt="output" src="https://github.com/user-attachments/assets/62f8616d-1769-4690-af31-36372b326499" />



Grafik heatmap di atas menunjukkan nilai koefisien korelasi Pearson antar saluran warna (Red, Green, Blue) dari seluruh piksel dataset daun padi. Terlihat nilai korelasi yang sangat kuat mendekati angka 1 (misalnya antara Red dan Green sebesar 0.98, serta Green dan Blue sebesar 0.95). Tingginya korelasi ini mengindikasikan adanya redundansi informasi warna yang tinggi pada citra alami daun. Bagi model CNN, pola korelasi ini menegaskan bahwa fitur bentuk geometris, tekstur bercak, dan gradasi spasial akan menjadi pembeda yang jauh lebih krusial bagi model dalam mengenali penyakit dibandingkan sekadar informasi warna murni individual.

<img width="898" height="770" alt="output" src="https://github.com/user-attachments/assets/d7f54aa5-550d-4c4a-a59d-c977585ef49a" />



Gambar di atas menampilkan grid visualisasi representatif dari masing-masing kelas penyakit daun padi yang digunakan untuk melatih model. Melalui sampel ini, kita dapat menganalisis karakteristik visual unik setiap kelas:

**Bacterial Blight**: Menunjukkan gejala garis-garis layu memanjang berwarna kuning pucat hingga putih di sepanjang tepi daun.

**Blast**: Ditandai dengan bercak berbentuk kornea/belah ketupat yang khas dengan bagian pusat berwarna abu-abu.

**Brown Spot**: Memperlihatkan bercak-bercak bulat atau oval kecil berwarna cokelat pekat yang tersebar merata di permukaan daun.

**Healthy**: Menampilkan daun padi sehat dengan warna hijau segar yang homogen tanpa bercak ataupun cacat fisik.

# 4. Exploratory Data Analysis (EDA)

Proses **Exploratory Data Analysis (EDA)** dilakukan untuk memahami karakteristik dataset sebelum memasuki tahap pemodelan. Tahapan EDA pada penelitian ini meliputi analisis distribusi kelas, visualisasi sampel citra, serta analisis struktur dataset.

## 4.1 Analisis Distribusi Kelas

Analisis distribusi kelas dilakukan dengan menghitung jumlah citra pada masing-masing kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**. Tujuan analisis ini adalah untuk mengetahui keseimbangan jumlah data yang digunakan pada proses pelatihan sehingga dapat meminimalkan kemungkinan model menjadi bias terhadap salah satu kelas.

<p align="center">
<img src="Gambar_Distribusi.png" width="650">
</p>

**Gambar 4.1 Distribusi Jumlah Citra pada Setiap Kelas**

Grafik batang di atas menunjukkan jumlah citra pada masing-masing kelas yang digunakan dalam penelitian. Berdasarkan hasil visualisasi, distribusi data antar kelas relatif seimbang sehingga dataset dinilai cukup baik untuk digunakan pada proses pelatihan model **CNN** dan **MobileNetV2**. Keseimbangan jumlah data tersebut membantu model mempelajari karakteristik setiap kelas secara lebih merata dan mengurangi kemungkinan terjadinya bias klasifikasi.

---

## 4.2 Visualisasi Sampel Citra

Visualisasi sampel citra dilakukan untuk melihat karakteristik visual dari setiap kelas penyakit daun tomat. Tahap ini membantu memahami perbedaan pola bercak, warna daun, serta tingkat kerusakan yang menjadi ciri khas masing-masing penyakit.

<p align="center">
<img src="Gambar_Sampel.png" width="800">
</p>

**Gambar 4.2 Contoh Citra Setiap Kelas**

Berdasarkan gambar di atas, setiap kelas memiliki karakteristik visual yang berbeda.

- **Healthy** menunjukkan daun berwarna hijau segar tanpa adanya bercak maupun kerusakan pada permukaan daun.
- **Early Blight** ditandai dengan munculnya bercak berwarna cokelat yang membentuk pola lingkaran konsentris pada permukaan daun.
- **Late Blight** memiliki bercak berwarna cokelat kehitaman yang cenderung lebih luas dan tidak beraturan sehingga menyebabkan kerusakan daun lebih parah dibandingkan Early Blight.

Perbedaan karakteristik visual tersebut menjadi informasi penting yang dipelajari oleh model Deep Learning untuk membedakan setiap kelas penyakit secara otomatis.

---

## 4.3 Analisis Struktur Dataset

Dataset yang digunakan telah disusun ke dalam dua folder utama, yaitu **train** dan **valid**. Folder **train** digunakan sebagai data pelatihan model, sedangkan folder **valid** digunakan sebagai data validasi selama proses training. Masing-masing folder terdiri atas tiga subfolder yang merepresentasikan kelas **Healthy**, **Early Blight**, dan **Late Blight**. Struktur dataset tersebut memungkinkan proses pemuatan data dilakukan secara otomatis menggunakan fungsi `flow_from_directory()` pada TensorFlow/Keras sehingga mempermudah proses pelatihan model.

# 5. Data Preparation

Tahap **Data Preparation** dilakukan untuk mempersiapkan dataset sebelum digunakan dalam proses pelatihan model **Convolutional Neural Network (CNN)** dan **MobileNetV2**. Pada penelitian ini, dataset telah dipisahkan ke dalam dua subset, yaitu **Training Set** dan **Validation Set**, sehingga tidak diperlukan proses pembagian data secara manual. Selanjutnya dilakukan beberapa tahapan praproses agar data memiliki format yang sesuai dengan kebutuhan model.

## Resize Citra

Seluruh citra diubah ukurannya menjadi **128 × 128 piksel** menggunakan parameter `target_size=(128,128)`. Proses ini bertujuan untuk menyeragamkan ukuran seluruh gambar sehingga dapat diproses oleh kedua model dengan lebih efisien serta mempercepat proses pelatihan.

## Data Augmentation

Untuk meningkatkan variasi data pelatihan dan mengurangi risiko **overfitting**, dilakukan proses augmentasi menggunakan **ImageDataGenerator**. Teknik augmentasi yang diterapkan meliputi:

- **Rotation Range (20°)** untuk memutar gambar secara acak.
- **Width Shift Range (0.2)** untuk menggeser gambar secara horizontal.
- **Height Shift Range (0.2)** untuk menggeser gambar secara vertikal.
- **Shear Range (0.2)** untuk memberikan transformasi shear pada gambar.
- **Zoom Range (0.2)** untuk memperbesar atau memperkecil gambar secara acak.
- **Horizontal Flip** untuk membalik gambar secara horizontal.
- **Fill Mode ("nearest")** untuk mengisi area kosong setelah transformasi.

Sementara itu, pada **Validation Set** hanya dilakukan proses **rescaling** tanpa augmentasi agar hasil evaluasi mencerminkan performa model terhadap data asli.

## Normalisasi Data

Normalisasi dilakukan menggunakan parameter **`rescale=1./255`**, sehingga seluruh nilai piksel yang semula berada pada rentang **0–255** diubah menjadi **0–1**. Langkah ini bertujuan untuk mempercepat proses pembelajaran model dan meningkatkan stabilitas selama proses training.

## Pembentukan Data Generator

Dataset dimuat menggunakan fungsi **`flow_from_directory()`** yang secara otomatis membaca struktur folder serta memberikan label sesuai nama folder kelas. Pada penelitian ini digunakan **batch size sebesar 32** gambar untuk setiap iterasi proses pelatihan.

### Tabel 5.1 Konfigurasi Dataset

| Jenis Subset Data | Direktori | Batch Size | Ukuran Citra | Fungsi |
|-------------------|-----------|:----------:|:------------:|--------|
| Data Training | `train/` | 32 | 128 × 128 | Melatih model CNN dan MobileNetV2 |
| Data Validation | `valid/` | 32 | 128 × 128 | Mengevaluasi performa model selama proses pelatihan |

Tabel di atas menunjukkan pembagian dataset yang digunakan pada penelitian ini ke dalam dua subset utama, yaitu **Training** dan **Validation**. Subset **Training** digunakan sebagai data utama untuk melatih model dalam mempelajari karakteristik visual dari setiap kelas penyakit daun tomat. Sementara itu, subset **Validation** dimanfaatkan untuk mengevaluasi performa model pada setiap epoch pelatihan melalui pengukuran nilai **accuracy** dan **loss**, sehingga dapat memantau proses pembelajaran serta mengurangi risiko **overfitting**. Dengan pembagian tersebut, proses pelatihan dan evaluasi dapat dilakukan secara terpisah sehingga model yang dihasilkan memiliki kemampuan generalisasi yang lebih baik terhadap data baru.

## 6. Modeling

Sesuai dengan ketentuan tugas yang mewajibkan penggunaan **minimal dua algoritma**, pada proyek ini diimplementasikan dua model **Deep Learning**, yaitu **Convolutional Neural Network (CNN)** dan **MobileNetV2**. Kedua model dilatih menggunakan dataset yang sama dengan konfigurasi pelatihan yang seragam sehingga hasil evaluasi dapat dibandingkan secara objektif.

### Model 1: Convolutional Neural Network (CNN)

- **Alasan Pemilihan:** CNN merupakan salah satu arsitektur Deep Learning yang dirancang khusus untuk pengolahan citra. Model ini mampu mengekstraksi fitur secara otomatis melalui kombinasi lapisan konvolusi dan pooling sehingga efektif dalam mengenali pola visual pada daun tomat yang terinfeksi penyakit.

- **Implementasi Model:** Model CNN dibangun menggunakan beberapa lapisan **Conv2D**, **MaxPooling2D**, **Flatten**, **Dense**, dan **Dropout**. Lapisan konvolusi digunakan untuk mengekstraksi fitur citra, sedangkan Dropout diterapkan untuk mengurangi risiko **overfitting**. Lapisan keluaran menggunakan fungsi aktivasi **Softmax** untuk menghasilkan klasifikasi tiga kelas.

### Model 2: MobileNetV2

- **Alasan Pemilihan:** MobileNetV2 merupakan arsitektur Deep Learning berbasis **Transfer Learning** yang telah dilatih menggunakan dataset **ImageNet**. Model ini memiliki jumlah parameter yang lebih sedikit sehingga proses pelatihan lebih efisien namun tetap mampu menghasilkan performa klasifikasi yang tinggi.

- **Implementasi Model:** MobileNetV2 digunakan sebagai **feature extractor** dengan bobot awal **ImageNet**. Lapisan dasar (**base model**) dibekukan (`trainable = False`), kemudian ditambahkan lapisan **GlobalAveragePooling2D**, **Dropout**, dan **Dense** dengan fungsi aktivasi **Softmax** sebagai lapisan klasifikasi akhir.

### Tabel 6.1 Perbandingan Konfigurasi Model

| Komponen Parameter | Model 1: CNN | Model 2: MobileNetV2 |
| :--- | :--- | :--- |
| **Arsitektur Model** | CNN Sequential | MobileNetV2 (ImageNet) |
| **Ukuran Input** | 128 × 128 × 3 piksel | 128 × 128 × 3 piksel |
| **Lapisan Keluaran** | Dense (3 Unit, Softmax) | Dense (3 Unit, Softmax) |
| **Fungsi Loss** | Categorical Crossentropy | Categorical Crossentropy |
| **Optimizer** | Adam (Learning Rate = 0.001) | Adam (Learning Rate = 0.001) |
| **Epoch** | 10 | 10 |
| **Batch Size** | 32 | 32 |

---

Tabel di atas menunjukkan bahwa kedua model dilatih menggunakan konfigurasi pelatihan yang sama, meliputi ukuran citra **128 × 128 piksel**, fungsi **Categorical Crossentropy**, **Adam Optimizer** dengan **learning rate 0.001**, **batch size 32**, serta **10 epoch** pelatihan. Penyamaan konfigurasi tersebut bertujuan agar proses perbandingan kedua model berlangsung secara adil (**fair comparison**), sehingga perbedaan performa yang diperoleh benar-benar dipengaruhi oleh karakteristik masing-masing arsitektur, bukan oleh perbedaan parameter pelatihan. Dengan demikian, hasil evaluasi dapat digunakan untuk menentukan model yang paling efektif dalam melakukan klasifikasi penyakit daun tomat.

# 7. Evaluation & Perbandingan Model

Evaluasi model dilakukan menggunakan **Validation Set** yang tidak digunakan secara langsung dalam proses pembelajaran model. Kedua model dievaluasi menggunakan beberapa metrik, yaitu **Accuracy**, **Precision**, **Recall**, **F1-Score**, serta **Confusion Matrix** untuk mengetahui kemampuan model dalam mengklasifikasikan citra daun tomat ke dalam tiga kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**.

## 1. Tabel Komparasi Metrik

| Nama Model | Accuracy | Precision | Recall | F1-Score | 
| :--- | :---: | :---: | :---: | :---: |
| **CNN** | 92.05% | 0.923 | 0.921 | 0.897 | 
| **MobileNetV2** | 96.52% | 0.773 | 0.762 | 0.751 | 

Tabel di atas menyajikan hasil evaluasi kedua model berdasarkan metrik **Accuracy**, **Precision**, **Recall**, dan **F1-Score**. Nilai-nilai tersebut digunakan sebagai dasar untuk membandingkan performa model **CNN** dan **MobileNetV2** dalam melakukan klasifikasi penyakit daun tomat. Model yang memperoleh nilai evaluasi lebih tinggi menunjukkan kemampuan klasifikasi yang lebih baik terhadap data validasi.

## 2. Analisis Grafik Perbandingan

Berdasarkan grafik perbandingan hasil pelatihan, kedua model menunjukkan peningkatan nilai **accuracy** dan penurunan **loss** selama proses training. Selain itu, dilakukan visualisasi grafik perbandingan akurasi akhir antara **CNN** dan **MobileNetV2** sehingga perbedaan performa kedua model dapat diamati dengan lebih jelas. Model dengan nilai akurasi lebih tinggi dapat dianggap memiliki kemampuan klasifikasi yang lebih baik pada dataset yang digunakan.

> **Gambar 7.1 Perbandingan Accuracy CNN dan MobileNetV2**

*(Masukkan grafik batang hasil notebook di sini.)*

## 3. Analisis Confusion Matrix

Confusion Matrix digunakan untuk melihat jumlah prediksi yang benar maupun salah pada masing-masing kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**. Nilai yang berada pada diagonal utama menunjukkan jumlah citra yang berhasil diklasifikasikan dengan benar, sedangkan nilai di luar diagonal menunjukkan kesalahan klasifikasi yang dilakukan oleh model.

Melalui Confusion Matrix dapat diketahui kelas mana yang memiliki tingkat kesalahan prediksi paling tinggi serta pasangan kelas yang sering mengalami kekeliruan. Informasi tersebut menjadi dasar dalam mengevaluasi kemampuan model dalam membedakan karakteristik visual dari masing-masing penyakit daun tomat.

> **Gambar 7.2 Confusion Matrix CNN**

*(Masukkan hasil Confusion Matrix CNN.)*

> **Gambar 7.3 Confusion Matrix MobileNetV2**

*(Masukkan hasil Confusion Matrix MobileNetV2.)*

---

# 8. Kesimpulan dan Rekomendasi

## Kesimpulan Hasil

1. **Ketercapaian Target:** Proyek ini berhasil mengembangkan dua model Deep Learning, yaitu **Convolutional Neural Network (CNN)** dan **MobileNetV2**, untuk melakukan klasifikasi penyakit daun tomat ke dalam tiga kelas, yaitu **Healthy**, **Early Blight**, dan **Late Blight**. Kedua model mampu melakukan proses klasifikasi dengan baik berdasarkan hasil evaluasi yang diperoleh.

2. **Perbandingan Model:** Berdasarkan hasil evaluasi, dilakukan perbandingan performa antara model **CNN** dan **MobileNetV2** menggunakan metrik **Accuracy**, **Loss**, **Precision**, **Recall**, **F1-Score**, serta **Confusion Matrix**. Model dengan nilai evaluasi terbaik dipilih sebagai model yang paling sesuai untuk digunakan pada proses klasifikasi penyakit daun tomat.

3. **Keterbatasan Penelitian:** Performa model masih dipengaruhi oleh kualitas citra yang digunakan, seperti pencahayaan, sudut pengambilan gambar, latar belakang, serta kondisi daun yang memiliki gejala penyakit yang hampir serupa. Faktor-faktor tersebut dapat menyebabkan terjadinya kesalahan klasifikasi pada beberapa citra.

## Rekomendasi Pengembangan

- **Penambahan Dataset:** Menambahkan jumlah data latih dengan variasi kondisi pencahayaan, sudut pengambilan gambar, dan kondisi lingkungan agar model memiliki kemampuan generalisasi yang lebih baik.

- **Pengembangan Model:** Mencoba arsitektur Deep Learning lain seperti **EfficientNet**, **ResNet**, atau **DenseNet** sebagai pembanding untuk memperoleh performa klasifikasi yang lebih tinggi.

- **Implementasi Aplikasi:** Mengembangkan model terbaik ke dalam aplikasi berbasis web atau perangkat mobile sehingga dapat digunakan oleh petani sebagai sistem deteksi penyakit daun tomat secara praktis dan real-time.

---

# 9. Referensi

1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv preprint arXiv:1704.04861.

2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520.

3. Hughes, D. P., & Salathé, M. (2015). *An Open Access Repository of Images on Plant Health to Enable the Development of Mobile Disease Diagnostics*. arXiv preprint arXiv:1511.08060.

4. TensorFlow. (2026). *Keras API Documentation*. https://www.tensorflow.org/api_docs

5. Kaggle. (2026). *PlantVillage Dataset*. https://www.kaggle.com/datasets/arjuntejaswi/plant-village

6. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using Deep Learning for Image-Based Plant Disease Detection*. Frontiers in Plant Science, 7, 1419.

7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---
## 10. Lampiran 

