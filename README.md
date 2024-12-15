# Laporan Proyek Machine Learning - Ziyad Muhammad Adzin Azzufari

## Project Overview

Dalam era digital saat ini, sistem rekomendasi telah menjadi komponen penting dalam membantu pengguna menemukan konten yang relevan dan menarik di berbagai platform. Berbagai sistem rekomendasi diterapkan di industri yang berbeda, mulai dari e-commerce hingga platform streaming. Tujuan utama dari proyek ini adalah untuk mengembangkan sistem rekomendasi buku menggunakan beberapa teknik pembelajaran mesin, dengan pendekatan yang berbeda untuk memberikan rekomendasi yang lebih personal dan relevan kepada pengguna. 

Proyek ini melibatkan pengembangan empat pendekatan utama:

1. **Popularity Based Recommender System**  
   Sistem ini memberikan rekomendasi berdasarkan popularitas buku di kalangan pengguna, dengan mempertimbangkan jumlah rating dan rating rata-rata sebagai indikator popularitas buku.

2. **Collaborative Filtering (Item-Based Filtering)**  
   Sistem ini memanfaatkan informasi dari pengguna lain yang memiliki preferensi serupa, untuk memberikan rekomendasi buku berdasarkan kesamaan item yang telah dinikmati oleh pengguna sebelumnya.

3. **Content-Based Filtering Berdasarkan Popularitas Judul Buku**  
   Pendekatan ini merekomendasikan buku berdasarkan kesamaan judul atau konten buku, dengan menggunakan teknik TF-IDF untuk menghitung kemiripan antar buku berdasarkan kata-kata kunci dalam judul buku.

4. **Content-Based Filtering Berdasarkan Rata-rata Rating Buku**  
   Pendekatan ini menggabungkan kesamaan buku berdasarkan konten dengan mempertimbangkan rating yang diberikan oleh pengguna, sehingga memberikan rekomendasi yang tidak hanya relevan dari segi isi buku tetapi juga berdasarkan kualitas atau rating buku tersebut.

Keempat pendekatan ini bertujuan untuk memberikan rekomendasi buku yang lebih baik dan lebih personal dengan memanfaatkan berbagai faktor, termasuk popularitas, kesamaan konten, dan rating pengguna.

## Business Understanding

### Problem Statements

Dalam proyek ini, terdapat beberapa pertanyaan utama yang ingin dijawab terkait sistem rekomendasi buku:

1. **Bagaimana cara merekomendasikan buku yang populer kepada pengguna berdasarkan jumlah rating dan rating rata-rata?**
   - Fokus pada pemilihan buku dengan rating tertinggi dan jumlah rating yang cukup untuk memastikan kualitas rekomendasi.
   
2. **Bagaimana membuat sistem rekomendasi berbasis Collaborative Filtering (Item-Based Filtering) yang memanfaatkan kesamaan antar buku?**
   - Menyusun model yang bisa memberikan rekomendasi berdasarkan rating pengguna dan kesamaan antar buku.
   
3. **Bagaimana menggunakan teknik Content-Based Filtering untuk merekomendasikan buku berdasarkan judul dan popularitasnya?**
   - Memanfaatkan fitur seperti judul buku, TF-IDF, dan kesamaan konten antar buku untuk memberikan rekomendasi.

4. **Bagaimana meningkatkan kualitas rekomendasi dengan mempertimbangkan rata-rata rating dan kesamaan konten?**
   - Menggabungkan informasi rating dan kesamaan konten untuk membuat rekomendasi yang lebih relevan.

### Goals

Tujuan utama dari proyek ini adalah untuk mengembangkan sistem rekomendasi buku yang dapat:

1. **Menghasilkan rekomendasi buku berdasarkan popularitas** dengan menggunakan rating dan jumlah rating.
   - Fokus pada buku yang mendapatkan banyak perhatian dan feedback dari pengguna.
   
2. **Menerapkan Collaborative Filtering untuk memberikan rekomendasi buku yang relevan** berdasarkan kesamaan buku yang telah dinilai oleh pengguna lain.

3. **Mengembangkan Content-Based Filtering dengan mempertimbangkan popularitas judul dan konten buku** seperti rating dan kesamaan teks.
   - Rekomendasi buku yang lebih personalisasi berdasarkan karakteristik buku yang relevan.

4. **Meningkatkan kualitas rekomendasi** dengan memanfaatkan kombinasi dari rating, jumlah rating, dan kesamaan konten untuk menghasilkan rekomendasi yang lebih baik.

### Solution Approach

Untuk mencapai tujuan tersebut, pendekatan berikut diusulkan:

1. **Popularity Based Recommender System**
   - Menganalisis buku yang memiliki jumlah rating lebih dari threshold tertentu dan memiliki rating rata-rata tinggi untuk memberikan rekomendasi buku yang populer.

2. **Collaborative Filtering (Item-Based Filtering)**
   - Menggunakan cosine similarity untuk mencari kesamaan antar buku berdasarkan rating yang diberikan oleh pengguna.
   - Menghasilkan rekomendasi buku berdasarkan buku yang serupa dengan buku yang telah dipilih atau dinilai oleh pengguna.

3. **Content-Based Filtering Berdasarkan Popularitas Judul Buku**
   - Menggunakan TF-IDF untuk menghitung kesamaan antar buku berdasarkan judul dan kata kunci dari buku tersebut.
   - Memberikan rekomendasi berdasarkan kemiripan judul buku yang telah diberi rating.

4. **Content-Based Filtering Berdasarkan Rata-rata Rating Buku**
   - Memilih buku yang memiliki rating rata-rata tinggi dan memberikan rekomendasi berdasarkan kesamaan judul buku tersebut menggunakan TF-IDF dan cosine similarity.

Dengan pendekatan ini, diharapkan dapat menciptakan sistem rekomendasi buku yang akurat, relevan, dan dapat meningkatkan pengalaman pengguna dalam menemukan buku-buku yang sesuai dengan preferensi mereka.

## Data Understanding

Dataset **Book-Crossing** terdiri dari tiga file utama yang menyimpan informasi terkait Users, Books, dan Ratings. Berikut adalah rincian tentang masing-masing file:

### 1. Users
File ini berisi informasi mengenai pengguna yang berinteraksi dengan buku dalam dataset. Setiap pengguna diidentifikasi dengan **User-ID** yang telah dianonimkan dan dipetakan ke dalam angka. Beberapa informasi tambahan mengenai pengguna, seperti **Lokasi** dan **Usia**, juga disediakan, namun kolom ini bisa saja berisi nilai **NULL** jika data tidak tersedia.

**Kolom dataset:**
- **User-ID**: ID unik yang mewakili pengguna (dianonimkan).
- **Lokasi**: Lokasi geografis pengguna, jika tersedia.
- **Usia**: Usia pengguna, jika tersedia.

### 2. Books
File ini mengandung informasi tentang buku-buku yang ada dalam dataset. Setiap buku diidentifikasi menggunakan **ISBN** (International Standard Book Number), dan buku yang memiliki ISBN tidak valid telah dihapus dari dataset.

Selain ISBN, informasi berbasis konten buku juga tersedia, seperti **Book-Title**, **Book-Author**, **Year-Of-Publication**, dan **Publisher**. Data ini diperoleh dari **Amazon Web Services**. Jika sebuah buku memiliki lebih dari satu penulis, hanya penulis pertama yang akan dicantumkan.

**Kolom dataset:**
- **ISBN**: Nomor ISBN unik yang mengidentifikasi buku.
- **Book-Title**: Judul buku.
- **Book-Author**: Nama penulis buku (hanya penulis pertama jika lebih dari satu).
- **Year-Of-Publication**: Tahun publikasi buku.
- **Publisher**: Penerbit buku.
- **Image-URL-S**: URL gambar sampul buku dalam ukuran kecil.
- **Image-URL-M**: URL gambar sampul buku dalam ukuran sedang.
- **Image-URL-L**: URL gambar sampul buku dalam ukuran besar.

### 3. Ratings
File ini berisi data penilaian yang diberikan oleh pengguna terhadap buku-buku. Penilaian dilakukan pada skala 1 hingga 10, dengan nilai yang lebih tinggi menunjukkan apresiasi yang lebih besar terhadap buku tersebut. Selain itu, ada juga nilai **0** yang menandakan penilaian implisit, di mana pengguna tidak memberikan rating eksplisit terhadap buku.

**Kolom dataset:**
- **User-ID**: ID pengguna yang memberikan rating.
- **ISBN**: ISBN buku yang diberi rating.
- **Book-Rating**: Rating yang diberikan oleh pengguna terhadap buku. Nilai ini berada dalam rentang 1-10 (rating eksplisit) atau 0 (rating implisit).

## Data Visualization

1. Top 10 Penerbit Berdasarkan Jumlah Buku

![alt text](image.png)

Visualisasi di atas menunjukkan distribusi jumlah buku berdasarkan tahun publikasi. Sebelum visualisasi ini dibuat, masalah terkait tahun publikasi yang tidak valid, seperti tahun outlier di masa depan (>2020) atau tahun 0, telah diatasi dengan mengganti nilai tersebut menjadi tahun 2002, yang merupakan tahun publikasi paling umum dalam dataset ini. Dengan demikian, data yang divisualisasikan mencerminkan distribusi yang lebih konsisten dan bebas dari outlier.

Melalui grafik ini, saya dapat mengidentifikasi tren penerbitan buku dari waktu ke waktu, seperti peningkatan jumlah buku pada tahun-tahun tertentu. Grafik ini memberikan wawasan awal mengenai pola publikasi buku dan dapat digunakan sebagai dasar untuk analisis lebih lanjut, misalnya apakah terdapat hubungan antara tren ini dengan variabel lain seperti genre atau penerbit tertentu.

2. Top 10 Penerbit Berdasarkan Jumlah Buku

![alt text](image-1.png)

- Harlequin merupakan penerbit buku terbanyak pertama dengan jumlah buku 7535 dengan perbedaan yang sangat signifikan dengan penerbit buku terbanyak kedua yaitu Silhouette dengan 4220 buku.

3. Top 10 Penulis Berdasarkan Jumlah Buku

![alt text](image-2.png)

- Penulis buku terbanyak pertama adalah Agatha Christie sebanyak 632 buku

4. Jumlah Buku Berdasarkan Tahun Publikasi dan Penerbit

![alt text](image-3.png)

- Berdasarkan hasil visualisasi diatas, penerbit terbanyak pertama yaitu Harlequin menerbitkan buku terbanyaknya pada tahun 2003

5. Jumlah Buku Berdasarkan Judul Buku (Top 10)

![alt text](image-4.png)

- Buku yang sama, meskipun ditulis oleh penulis yang sama, sering kali memiliki beberapa ISBN unik. Ini dapat terjadi karena buku tersebut diterbitkan oleh penerbit yang berbeda atau diterbitkan dalam tahun yang berbeda. Dalam konteks membangun sistem rekomendasi buku, hal ini menjadi penting untuk diperhatikan. Jika buku yang sama diidentifikasi dengan ISBN yang berbeda, ini bisa menyebabkan rekomendasi menjadi kurang akurat atau berulang. Oleh karena itu, nantinya mungkin saya perlu menyatukan ISBN untuk edisi-edisi buku yang sama agar sistem rekomendasi dapat bekerja lebih optimal. Namun, langkah ini akan dipertimbangkan lebih lanjut selama proses pengembangan sistem rekomendasi.

6. Distribusi Jumlah Rating Pengguna yang Kurang dari 50

![alt text](image-5.png)

- Histogram ini menunjukkan bahwa sebagian besar pengguna memberikan sedikit rating, khususnya pengguna dengan hanya 1 atau 2 rating. Pengguna dengan jumlah rating rendah ini mendominasi distribusi, sementara hanya sebagian kecil pengguna yang memberikan banyak rating. Pengguna yang memberikan sedikit rating mungkin tidak memberikan gambaran yang cukup kuat tentang preferensi mereka, yang menjadi perhatian penting dalam pengembangan sistem rekomendasi, karena rekomendasi yang diberikan berdasarkan rating terbatas dapat kurang akurat.

Hal ini menjadi jelas jika saya membuat histogram yang sama dengan batas untuk pengguna dengan minimal 1000 ratings.

![alt text](image-6.png)

7. Distribusi Rating Buku

![alt text](image-7.png)

Hasil visualisasi menunjukkan bahwa sebagian besar entri memiliki rating 0, yang dalam konteks ini diartikan sebagai rating implisit. Rating implisit berbeda dengan rating eksplisit karena rating 0 ini tidak menunjukkan penilaian yang eksplisit dari pengguna, melainkan sebuah interaksi (positif atau negatif) antara pengguna dan buku tersebut. Sebagai langkah pemodelan, saya hanya akan fokus pada rating eksplisit (nilai dari 1 hingga 10), dan oleh karena itu, entri dengan rating 0 akan dihapus.

Berikut merupakan hasil visualisasi Distribusi Rating Eksplisit Buku (1-10)

![alt text](image-8.png)

