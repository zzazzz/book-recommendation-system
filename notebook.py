# %% [markdown]
# ## Data Understanding

# %% [markdown]
# Isi Dataset
# Dataset Book-Crossing terdiri dari 3 file.
# 
# - Users
# 
# Berisi data pengguna. Perlu diperhatikan bahwa ID pengguna (User-ID) telah dianonimkan dan dipetakan ke dalam angka. Data demografi (Lokasi, Usia) disediakan jika tersedia. Jika tidak, kolom ini akan berisi nilai NULL.
# 
# - Books
# 
# Buku diidentifikasi berdasarkan ISBN masing-masing. ISBN yang tidak valid telah dihapus dari dataset. Selain itu, terdapat beberapa informasi berbasis konten seperti Book-Title, Book-Author, Year-Of-Publication, dan Publisher, yang diperoleh dari Amazon Web Services. Perlu dicatat bahwa jika ada beberapa penulis, hanya penulis pertama yang dicantumkan. URL yang mengarah ke gambar sampul buku juga disediakan dalam tiga ukuran berbeda (Image-URL-S, Image-URL-M, Image-URL-L), yaitu kecil, sedang, dan besar. URL ini mengarah ke situs Amazon.
# 
# - Ratings
# 
# Berisi informasi mengenai penilaian buku. Penilaian (Book-Rating) dapat berupa eksplisit, yang dinyatakan dalam skala 1-10 (dengan nilai lebih tinggi menunjukkan apresiasi lebih tinggi), atau implisit, yang dinyatakan dengan nilai 0.

# %%
import pandas as pd

books = pd.read_csv('Books.csv')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

# %%
books.head()

# %% [markdown]
# books dataset:
# 
# 1. ISBN: ISBN adalah singkatan dari International Standard Book Number. ISBN merupakan pengenal numerik unik yang digunakan untuk mengidentifikasi buku dan publikasi monograf lainnya. Setiap ISBN terdiri dari rangkaian digit yang secara unik mengidentifikasi edisi atau versi tertentu dari sebuah buku.
# Tujuan ISBN adalah menyediakan metode standar untuk mengidentifikasi dan mengkatalogkan buku. ISBN membantu penerbit, penjual buku, perpustakaan, dan organisasi lain untuk mengelola dan melacak buku dengan efisien. Dengan menggunakan ISBN, sebuah buku dapat dengan mudah ditemukan, dipesan, dan dijual secara internasional.
# 
# 2. Book-Title: Judul dari buku.
# 
# 3. Book-Author: Penulis buku.
# 
# 4. Year-Of-Publication: Tahun penerbitan buku.
# 
# 5. Publisher: Penerbit adalah organisasi atau perusahaan yang bertanggung jawab untuk memproduksi dan mendistribusikan buku kepada publik. Penerbit menangani berbagai aspek dari proses penerbitan buku, termasuk mendapatkan naskah, mengedit, merancang, mencetak, memasarkan, dan mendistribusikan buku.
# 
# 6. Image-URL-S: URL pendek untuk gambar sampul buku.
# 
# 7. Image-URL-M: URL ukuran sedang untuk gambar sampul buku.
# 
# 8. Image-URL-L: URL ukuran panjang untuk gambar sampul buku.

# %%
users.head()

# %% [markdown]
# users dataset:
# 
# 1. User-ID: ID pengguna dari pembaca/pelanggan yang memberikan ulasan untuk buku.
# 2. Location: Lokasi pengguna.
# 3. Age: Usia pengguna.

# %%
ratings.head()

# %% [markdown]
# ratings dataset:
# 
# 1. User-ID: ID pengguna dari pembaca/pelanggan yang memberikan ulasan untuk buku.
# 2. ISBN: Pengenal unik untuk buku yang telah diulas.
# 3. Book-Rating: Penilaian untuk buku tertentu.

# %% [markdown]
# ## Univariate Exploratory Data Analysis and Data Preparation

# %% [markdown]
# ### Books

# %%
books.columns

# %%
## Drop URL columns
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1, inplace=True)

# %% [markdown]
# Kode diatas menjelaskan membuang kolom yang tidak digunakan untuk analis lebih lanjut.

# %%
print("Books Data:", books.shape)

# %% [markdown]
# Total baris pada dataset books adalah 271360 dan total kolom adalah 5. 

# %%
## Checking for null values
books.isnull().sum()

# %% [markdown]
# Berdasarkan hasil diatas, terdapat nilai kosong sebanyak 2 pada kolom "Book-Author" dan "Publisher"

# %%
# Menampilkan semua baris dalam dataset Books di mana kolom 'Book-Author' bernilai kosong (NaN).
books[books['Book-Author'].isnull()]

# %%
# Menampilkan semua baris dalam dataset Books di mana kolom 'Publisher' bernilai kosong (NaN).
books[books['Publisher'].isnull()]

# %%
# Mengisi nilai kosong (NaN) pada kolom 'Book-Author' dengan string 'Other'.
books['Book-Author'] = books['Book-Author'].fillna('Other')

# Mengisi nilai kosong (NaN) pada kolom 'Publisher' dengan string 'Other'.
books['Publisher'] = books['Publisher'].fillna('Other')

# %%
# books[books['Book-Author']=='Other']

# %%
# books[books['Publisher']=='Other']

# %%
## Checking for column Year-of-publication
books['Year-Of-Publication'].unique()

# %% [markdown]
# Saya melihat bahwa kolom Year-Of-Publication memiliki tipe data object, dan terdapat beberapa masalah, seperti nilai nol (0) serta kesalahan lainnya, misalnya nama-nama seperti "Gallimard" dan "DK Publishing Inc" yang secara keliru dimasukkan ke dalam kolom tahun penerbitan. Oleh karena itu, saya akan memperbaiki kesalahan tersebut.

# %% [markdown]
# Mendeteksi kesalahan pada kolom Year-Of-Publication:
# 
# Syntax dibawah digunakan untuk menemukan entri di mana kolom Year-Of-Publication memiliki nilai 'DK Publishing Inc', yang seharusnya merupakan nilai dari kolom Publisher.

# %%
books[books['Year-Of-Publication']=='DK Publishing Inc']

# %% [markdown]
# Memeriksa informasi tambahan untuk ISBN tertentu:
# 
# Syntax dibawah digunakan untuk melihat judul buku (Book-Title) yang terkait dengan ISBN 078946697X.

# %%
books[books['ISBN'] == '078946697X']['Book-Title'].values[0]

# %% [markdown]
# hasil dari syntax diatas datanya tidak terstruktur dengan baik karena berisi nama penulis di dalamnya.

# %% [markdown]
# Kesalahan dalam entri dengan ISBN 078946697X diperbaiki dengan menetapkan ulang nilai kolom yang relevan:
# 
# - Publisher diatur menjadi 'DK Publishing Inc'.
# - Book-Author diatur menjadi 'Michael Teitelbaum'.
# - Year-Of-Publication diatur menjadi 2000.
# - Book-Title diperbaiki menjadi judul yang benar tanpa nama penulis: 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'.

# %%
books.loc[books['ISBN'] == '078946697X', 'Publisher'] = 'DK Publishing Inc'
books.loc[books['ISBN'] == '078946697X', 'Book-Author'] = 'Michael Teitelbaum'
books.loc[books['ISBN'] == '078946697X', 'Year-Of-Publication'] = 2000
books.loc[books['ISBN'] == '078946697X', 'Book-Title'] = 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)'

# %% [markdown]
# Setelah dilakukan perbaikan, syntax dibawah ini digunakan untuk memverifikasi dan menampilkan kembali baris dalam dataset Books yang memiliki ISBN 078946697X. Tujuannya adalah untuk memastikan bahwa data yang telah diperbaiki, seperti Book-Title, Book-Author, Year-Of-Publication, dan Publisher, sudah sesuai dan tidak ada lagi kesalahan dalam entri tersebut setelah dilakukan pembaruan. Kode ini memastikan bahwa entri dengan ISBN tersebut telah diperbaiki dan data konsisten.

# %%
# Menampilkan baris dalam dataset Books yang memiliki ISBN '078946697X' untuk memeriksa informasi buku terkait.
books[books['ISBN'] == '078946697X']

# %% [markdown]
# Memeriksa informasi tambahan untuk ISBN tertentu:
# 
# Syntax dibawah digunakan untuk melihat judul buku (Book-Title) yang terkait dengan ISBN 0789466953.

# %%
books[books['ISBN'] == '0789466953']['Book-Title'].values[0]

# %% [markdown]
# hasil dari syntax diatas datanya tidak terstruktur dengan baik karena berisi nama penulis di dalamnya.

# %% [markdown]
# Kesalahan dalam entri dengan ISBN 078946697X diperbaiki dengan menetapkan ulang nilai kolom yang relevan:
# 
# - Publisher diatur menjadi 'DK Publishing Inc'.
# - Book-Author diatur menjadi 'James Buckley'.
# - Year-Of-Publication diatur menjadi 2000.
# - Book-Title diperbaiki menjadi judul yang benar tanpa nama penulis: 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'.

# %%
books.loc[books['ISBN'] == '0789466953', 'Publisher'] = 'DK Publishing Inc'
books.loc[books['ISBN'] == '0789466953', 'Book-Author'] = 'James Buckley'
books.loc[books['ISBN'] == '0789466953', 'Year-Of-Publication'] = 2000
books.loc[books['ISBN'] == '0789466953', 'Book-Title'] = 'DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)'

# %% [markdown]
# Setelah dilakukan perbaikan, syntax dibawah ini digunakan untuk memverifikasi dan menampilkan kembali baris dalam dataset Books yang memiliki ISBN 0789466953. Tujuannya adalah untuk memastikan bahwa data yang telah diperbaiki, seperti Book-Title, Book-Author, Year-Of-Publication, dan Publisher, sudah sesuai dan tidak ada lagi kesalahan dalam entri tersebut setelah dilakukan pembaruan. Kode ini memastikan bahwa entri dengan ISBN tersebut telah diperbaiki dan data konsisten.

# %%
books[books['ISBN'] == '0789466953']

# %% [markdown]
# Mendeteksi kesalahan pada kolom Year-Of-Publication:
# 
# Syntax dibawah digunakan untuk menemukan entri di mana kolom Year-Of-Publication memiliki nilai 'Gallimard', yang seharusnya merupakan nilai dari kolom Publisher.

# %%
books[books['Year-Of-Publication']=='Gallimard']

# %% [markdown]
# Memeriksa informasi tambahan untuk ISBN tertentu:
# 
# Syntax dibawah digunakan untuk melihat judul buku (Book-Title) yang terkait dengan ISBN 070426769.

# %%
books[books['ISBN'] == '2070426769']['Book-Title'].values[0]

# %% [markdown]
# Hasil dari syntax diatas datanya tidak terstruktur dengan baik karena berisi nama penulis di dalamnya.

# %% [markdown]
# Kesalahan dalam entri dengan ISBN 078946697X diperbaiki dengan menetapkan ulang nilai kolom yang relevan:
# 
# - Publisher diatur menjadi 'Gallimard'.
# - Book-Author diatur menjadi 'Jean-Marie Gustave Le ClÃ?Â©zio'.
# - Year-Of-Publication diatur menjadi 2003.
# - Book-Title diperbaiki menjadi judul yang benar tanpa nama penulis: 'Peuple du ciel - Suivi de Les bergers'.

# %%
books.loc[books['ISBN'] == '2070426769', 'Publisher'] = 'Gallimard'
books.loc[books['ISBN'] == '2070426769', 'Book-Author'] = 'Jean-Marie Gustave Le ClÃ?Â©zio'
books.loc[books['ISBN'] == '2070426769', 'Year-Of-Publication'] = 2003
books.loc[books['ISBN'] == '2070426769', 'Book-Title'] = 'Peuple du ciel - Suivi de Les bergers'

# %% [markdown]
# Setelah dilakukan perbaikan, syntax dibawah ini digunakan untuk memverifikasi dan menampilkan kembali baris dalam dataset Books yang memiliki ISBN 2070426769. Tujuannya adalah untuk memastikan bahwa data yang telah diperbaiki, seperti Book-Title, Book-Author, Year-Of-Publication, dan Publisher, sudah sesuai dan tidak ada lagi kesalahan dalam entri tersebut setelah dilakukan pembaruan. Kode ini memastikan bahwa entri dengan ISBN tersebut telah diperbaiki dan data konsisten.

# %%
books[books['ISBN'] == '2070426769']

# %% [markdown]
# Menentukan Tahun Paling Umum:

# %%
most_frequent = books['Year-Of-Publication'].value_counts()
most_common_year = most_frequent[most_frequent == most_frequent.max()].index.tolist()
most_common_year

# %% [markdown]
# Mengubah Tipe Data Kolom Tahun menjadi integer:

# %%
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
print(sorted(books['Year-Of-Publication'].unique()))

# %% [markdown]
# Menghitung Tahun yang Tidak Valid:

# %%
# Menghitung publikasi yang Year-Of-Publication lebih dari 2020
above_2020 = books[books['Year-Of-Publication'] > 2020]
count_above_2020 = above_2020.shape[0]

# Menghitung publikasi yang Year-Of-Publication sama dengan 0
equal_to_0 = books[books['Year-Of-Publication'] == 0]
count_equal_to_0 = equal_to_0.shape[0]

# Total jumlah publikasi
total_count = books.shape[0]

# Menampilkan hasil
print(f"Jumlah publikasi dengan Year-Of-Publication > 2020: {count_above_2020}")
print(f"Jumlah publikasi dengan Year-Of-Publication = 0: {count_equal_to_0}")
print(f"Total publikasi: {total_count}")

# %% [markdown]
# Memperbaiki Tahun yang Tidak Valid:

# %%
books.loc[(books['Year-Of-Publication'] > 2020) | (books['Year-Of-Publication'] == 0), 'Year-Of-Publication'] = 2002

# %% [markdown]
# Memperbaiki entri tersebut dengan mengganti nilai tahun publikasi menjadi tahun yang paling umum, yaitu 2002, untuk memastikan data lebih konsisten dan realistis.

# %%
print("The number of 0s in Year-Of-Publication column after using mode is: " + str((books['Year-Of-Publication'] == 0).sum()))

# %%
print("The number of more than 2020s in Year-Of-Publication column after using mode is: " + str((books['Year-Of-Publication'] > 2020).sum()))

# %%
books.isna().sum()

# %%
books

# %% [markdown]
# Pembersihan Data pada Kolom Publisher
# 
# Pada langkah ini, saya melakukan pembersihan data pada kolom Publisher dalam dataset Books. Terkadang, data yang diambil dari sumber web atau layanan seperti Amazon mengandung karakter khusus yang di-encode dalam format HTML. Salah satu contohnya adalah simbol ampersand (&) yang di-encode sebagai &amp;.
# 
# Untuk memastikan bahwa data Publisher konsisten dan mudah dibaca, saya mengganti semua kemunculan &amp; dengan simbol ampersand biasa (&). Proses ini tidak hanya meningkatkan keterbacaan data tetapi juga mencegah potensi kesalahan dalam analisis atau visualisasi data selanjutnya.

# %%
books.Publisher = books.Publisher.str.replace('&amp;', '&', regex=False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plot
sns.set(style="whitegrid")

# Plotting the distribution of Year-Of-Publication
plt.figure(figsize=(10,6))
sns.countplot(data=books, x='Year-Of-Publication', palette='viridis')
plt.title('Distribusi Jumlah Buku Berdasarkan Tahun Publikasi', fontsize=16)
plt.xticks(rotation=90)  # Rotate the x-axis labels for better visibility
plt.xlabel('Tahun Publikasi')
plt.ylabel('Jumlah Buku')
plt.show()


# %% [markdown]
# Visualisasi di atas menunjukkan distribusi jumlah buku berdasarkan tahun publikasi. Sebelum visualisasi ini dibuat, masalah terkait tahun publikasi yang tidak valid, seperti tahun outlier di masa depan (>2020) atau tahun 0, telah diatasi dengan mengganti nilai tersebut menjadi tahun 2002, yang merupakan tahun publikasi paling umum dalam dataset ini. Dengan demikian, data yang divisualisasikan mencerminkan distribusi yang lebih konsisten dan bebas dari outlier.
# 
# Melalui grafik ini, saya dapat mengidentifikasi tren penerbitan buku dari waktu ke waktu, seperti peningkatan jumlah buku pada tahun-tahun tertentu. Grafik ini memberikan wawasan awal mengenai pola publikasi buku dan dapat digunakan sebagai dasar untuk analisis lebih lanjut, misalnya apakah terdapat hubungan antara tren ini dengan variabel lain seperti genre atau penerbit tertentu.

# %%
# Plotting the top 10 publishers
top_publishers = books['Publisher'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_publishers.values, y=top_publishers.index, palette='coolwarm')
plt.title('Top 10 Penerbit Berdasarkan Jumlah Buku', fontsize=16)
plt.xlabel('Jumlah Buku')
plt.ylabel('Penerbit')
plt.show()

# %%
# Plotting the top 10 authors
top_authors = books['Book-Author'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_authors.values, y=top_authors.index, palette='Set2')
plt.title('Top 10 Penulis Berdasarkan Jumlah Buku', fontsize=16)
plt.xlabel('Jumlah Buku')
plt.ylabel('Penulis')
plt.show()


# %%
# Menyaring 10 penerbit dengan jumlah buku terbanyak
top_publishers = books['Publisher'].value_counts().head(5).index
filtered_books = books[books['Publisher'].isin(top_publishers)]

# Menyaring data dengan tahun publikasi antara 1980 dan 2020
filtered_books = filtered_books[(filtered_books['Year-Of-Publication'] >= 1980) & 
                                 (filtered_books['Year-Of-Publication'] <= 2020)]

# Plotting the number of books published by publisher and year
plt.figure(figsize=(14,8))
sns.countplot(data=filtered_books, x='Year-Of-Publication', hue='Publisher', palette='tab20', dodge=True)
plt.title('Jumlah Buku Berdasarkan Tahun Publikasi dan Penerbit', fontsize=16)
plt.xticks(rotation=90)
plt.xlabel('Tahun Publikasi')
plt.ylabel('Jumlah Buku')
plt.show()

# %%
# Count unique publishers and authors
unique_publishers = books['Publisher'].nunique()
unique_authors = books['Book-Author'].nunique()

# Display the counts
print(f"Jumlah Penerbit Unik: {unique_publishers}")
print(f"Jumlah Penulis Unik: {unique_authors}")

# %%
# Menyaring 10 judul buku dengan jumlah terbanyak
top_books = books['Book-Title'].value_counts().head(10).index
filtered_books_by_title = books[books['Book-Title'].isin(top_books)]

# Plotting the number of books by book title
plt.figure(figsize=(14,8))
sns.countplot(data=filtered_books_by_title, y='Book-Title', palette='Set2', order=top_books)
plt.title('Jumlah Buku Berdasarkan Judul Buku (Top 10)', fontsize=16)
plt.xlabel('Jumlah Buku')
plt.ylabel('Judul Buku')
plt.show()


# %%
books[books['Book-Title']=='Little Women']

# %% [markdown]
# Buku yang sama, meskipun ditulis oleh penulis yang sama, sering kali memiliki beberapa ISBN unik. Ini dapat terjadi karena buku tersebut diterbitkan oleh penerbit yang berbeda atau diterbitkan dalam tahun yang berbeda. Dalam konteks membangun sistem rekomendasi buku, hal ini menjadi penting untuk diperhatikan. Jika buku yang sama diidentifikasi dengan ISBN yang berbeda, ini bisa menyebabkan rekomendasi menjadi kurang akurat atau berulang. Oleh karena itu, nantinya mungkin saya perlu menyatukan ISBN untuk edisi-edisi buku yang sama agar sistem rekomendasi dapat bekerja lebih optimal. Namun, langkah ini akan dipertimbangkan lebih lanjut selama proses pengembangan sistem rekomendasi.

# %% [markdown]
# ### Ratings

# %%
print("Books-ratings:", ratings.shape)

# %% [markdown]
# Total baris pada dataset books adalah 1149780 dan total kolom adalah 3. 

# %%
## Checking for null values
ratings.isnull().sum() 

# %%
# Checking of duplicates 
ratings.duplicated().sum()

# %%
ratings.info()

# %% [markdown]
# Sintak ratings.info() memberikan gambaran tentang dataset ratings yang memiliki 1.149.780 entri dan 3 kolom: User-ID (int64), ISBN (object), dan Book-Rating (int64). 

# %%
ratings['Book-Rating'].unique()

# %% [markdown]
# Sintak ratings['Book-Rating'].unique() menampilkan nilai unik yang terdapat pada kolom Book-Rating dalam dataset ratings. Nilai yang muncul adalah angka antara 0 hingga 10, yang menunjukkan berbagai tingkat penilaian buku oleh pengguna. Nilai 0 biasanya mengindikasikan rating implisit atau ketidakadaan penilaian.

# %%
# Visualisasi jumlah rating per buku (Top 10 Buku dengan Rating Terbanyak)
top_rated_books = ratings.groupby('ISBN').size().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
ax = top_rated_books.plot(kind='bar', color='teal')
plt.title('Top 10 Buku dengan Rating Terbanyak', fontsize=16)
plt.xlabel('ISBN Buku')
plt.ylabel('Jumlah Rating')
plt.xticks(rotation=45)

# Menambahkan label total di atas setiap batang
for i, v in enumerate(top_rated_books):
    ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=10)

plt.show()


# %%
# Visualisasi distribusi User-ID (jumlah rating per user)
user_ratings_count = ratings.groupby('User-ID').size().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
ax = user_ratings_count.plot(kind='bar', color='orange')
plt.title('Top 10 Pengguna dengan Rating Terbanyak', fontsize=16)
plt.xlabel('User-ID')
plt.ylabel('Jumlah Rating')

# Menambahkan label total di atas setiap batang
for i, v in enumerate(user_ratings_count):
    ax.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=10)

plt.show()


# %%
# Visualisasi distribusi User-ID (jumlah rating per user)
user_ratings_count = ratings.groupby('User-ID').size().sort_values(ascending=False)

# Menyaring pengguna yang memberikan rating kurang dari 50
user_ratings_less_than_50 = user_ratings_count[user_ratings_count < 50]

# Visualisasi distribusi jumlah rating per pengguna yang kurang dari 50 dalam bentuk histogram
plt.figure(figsize=(12,6))
plt.hist(user_ratings_less_than_50, bins=30, color='lightblue', edgecolor='black')
plt.title('Distribusi Jumlah Rating Pengguna yang Kurang dari 50', fontsize=16)
plt.xlabel('Jumlah Rating')
plt.ylabel('Jumlah Pengguna')
plt.grid(True)
plt.show()

# %% [markdown]
# Pada sintaks ini, pertama-tama dilakukan pengelompokan data berdasarkan User-ID untuk menghitung jumlah rating yang diberikan oleh setiap pengguna. Kemudian, data yang diperoleh disortir berdasarkan jumlah rating dari yang terbanyak hingga yang paling sedikit.
# 
# Setelah itu, pengguna yang memberikan rating kurang dari 50 kali dipilih untuk dianalisis lebih lanjut. Visualisasi histogram kemudian dibuat untuk menunjukkan distribusi jumlah rating yang diberikan oleh pengguna yang memberikan kurang dari 50 rating.
# 
# Histogram ini menunjukkan bahwa sebagian besar pengguna memberikan sedikit rating, khususnya pengguna dengan hanya 1 atau 2 rating. Pengguna dengan jumlah rating rendah ini mendominasi distribusi, sementara hanya sebagian kecil pengguna yang memberikan banyak rating. Pengguna yang memberikan sedikit rating mungkin tidak memberikan gambaran yang cukup kuat tentang preferensi mereka, yang menjadi perhatian penting dalam pengembangan sistem rekomendasi, karena rekomendasi yang diberikan berdasarkan rating terbatas dapat kurang akurat.

# %% [markdown]
# Hal ini menjadi jelas jika saya membuat histogram yang sama dengan batas untuk pengguna dengan minimal 1000 ratings.

# %%
# Menyaring pengguna yang memberikan rating lebih dari 1000
user_ratings_more_than_1000 = user_ratings_count[user_ratings_count > 1000]

# Visualisasi distribusi jumlah rating per pengguna yang kurang dari 50 dalam bentuk histogram
plt.figure(figsize=(12,6))
plt.hist(user_ratings_more_than_1000, bins=30, color='lightblue', edgecolor='black')
plt.title('Distribusi Jumlah Rating Pengguna yang Lebih dari 1000', fontsize=16)
plt.xlabel('Jumlah Rating')
plt.ylabel('Jumlah Pengguna')
plt.grid(True)
plt.show()

# %% [markdown]
# Pada sintaks ini, saya pertama-tama menganalisis distribusi nilai rating pada dataset dengan menghitung jumlah kemunculan setiap rating menggunakan value_counts() dan mengurutkannya berdasarkan nilai rating (dari yang terkecil hingga terbesar). Selanjutnya, distribusi rating tersebut divisualisasikan menggunakan grafik batang (bar plot).

# %%
# Distribusi Book-Rating
rating_counts = ratings['Book-Rating'].value_counts().sort_index()

plt.figure(figsize=(12,6))
rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribusi Rating Buku', fontsize=16)
plt.xlabel('Rating')
plt.ylabel('Jumlah Rating')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Menambahkan label total di atas setiap batang
for i, v in enumerate(rating_counts):
    plt.text(i, v + 100, str(v), ha='center', va='bottom', fontsize=10)

plt.show()


# %% [markdown]
# Hasil visualisasi menunjukkan bahwa sebagian besar entri memiliki rating 0, yang dalam konteks ini diartikan sebagai rating implisit. Rating implisit berbeda dengan rating eksplisit karena rating 0 ini tidak menunjukkan penilaian yang eksplisit dari pengguna, melainkan sebuah interaksi (positif atau negatif) antara pengguna dan buku tersebut. Sebagai langkah pemodelan, saya hanya akan fokus pada rating eksplisit (nilai dari 1 hingga 10), dan oleh karena itu, entri dengan rating 0 akan dihapus.
# 
# Selain itu, untuk memberikan informasi lebih lengkap pada grafik, jumlah rating di atas setiap batang grafik juga ditambahkan dengan menggunakan fungsi plt.text(), yang menunjukkan total jumlah rating untuk setiap nilai rating yang diberikan. Dengan demikian, saya dapat dengan mudah memahami distribusi rating dalam dataset.

# %% [markdown]
# Sintaks dibawah ini memfilter rating eksplisit (nilai selain 0) dari dataset dan menghitung distribusi jumlah rating untuk setiap nilai rating (1-10). Hasilnya divisualisasikan dalam bentuk grafik batang, dengan sumbu x menunjukkan rating dan sumbu y menunjukkan jumlah rating yang diberikan.

# %%
explicit_ratings = ratings[ratings['Book-Rating'] != 0]

# Visualize explicit ratings
explicit_rating_counts = explicit_ratings['Book-Rating'].value_counts().sort_index()

plt.figure(figsize=(12,6))
explicit_rating_counts.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title('Distribusi Rating Eksplisit Buku (1-10)', fontsize=16)
plt.xlabel('Rating Eksplisit')
plt.ylabel('Jumlah Rating')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding labels to the bars
for i, v in enumerate(explicit_rating_counts):
    plt.text(i, v + 100, str(v), ha='center', va='bottom', fontsize=10)

plt.show()

# %% [markdown]
# ### Users

# %%
print("Users Data:", users.shape)

# %% [markdown]
# Total baris pada dataset books adalah 278858 dan total kolom adalah 3.

# %%
## Checking for null values
users.isnull().sum() 

# %%
# Checking of duplicates 
users.duplicated().sum()

# %%
users.info()

# %% [markdown]
# Sintak ratings.info() memberikan gambaran tentang dataset ratings yang memiliki 278858 entri dan 3 kolom: User-ID (int64), Location (object), dan Age (float64). 

# %%
users[users['Age'].isnull()]

# %%
## Check for all values present in Age column
print(sorted(list(users['Age'].unique())))

# %% [markdown]
# Berdasarkan hasil pengecekan nilai yang terdapat pada kolom 'Age', terdapat beberapa nilai yang tidak realistis, seperti usia lebih dari 80 atau kurang dari 10 tahun. Nilai usia seperti ini bisa dianggap sebagai outlier yang perlu ditangani agar tidak mempengaruhi analisis atau model yang dikembangkan.
# 
# Untuk menangani outlier ini, langkah yang diambil adalah mengganti nilai usia yang lebih besar dari 80 dan lebih kecil dari 10 dengan nilai rata-rata usia yang valid. Pertama, data yang memiliki usia antara 10 hingga 80 diambil untuk menghitung rata-ratanya. Kemudian, usia yang lebih besar dari 80 atau lebih kecil dari 10 diganti dengan nilai rata-rata yang sudah dihitung tersebut. Selain itu, nilai yang kosong (null) pada kolom usia juga diisi dengan rata-rata tersebut, dan akhirnya kolom usia diubah menjadi tipe data integer untuk konsistensi.
# 
# Dengan cara ini, nilai yang tidak realistis atau hilang pada kolom usia dapat diperbaiki tanpa mengubah distribusi usia secara signifikan.

# %%
required = users[users['Age'] <= 80]
required = required[required['Age'] >= 10]

# %%
mean = round(required['Age'].mean())   
mean

# %%
users.loc[users['Age'] > 80, 'Age'] = mean    #outliers with age grater than 80 are substituted with mean 
users.loc[users['Age'] < 10, 'Age'] = mean    #outliers with age less than 10 years are substitued with mean
users['Age'] = users['Age'].fillna(mean)      #filling null values with mean
users['Age'] = users['Age'].astype(int)       #changing Datatype to int

# %%
users.isnull().sum()

# %%
## Check for all values present in Age column
print(sorted(list(users['Age'].unique())))

# %%
# Pisahkan kolom Location menjadi tiga kolom: City, Region, Country
split_location = users['Location'].str.split(',', expand=True)

# Isi nilai yang kosong (karena kurang dari 3 elemen) dengan NaN
split_location = split_location.fillna('')

# Berikan nama kolom City, Region, Country
users[['City', 'Region', 'Country']] = split_location.iloc[:, :3]

# Hapus spasi tambahan di awal dan akhir setiap kolom
users['City'] = users['City'].str.strip()
users['Region'] = users['Region'].str.strip()
users['Country'] = users['Country'].str.strip()

# Lihat hasilnya
users.drop(columns='Location', inplace=True)
users

# %% [markdown]
# Pada kode di atas, kolom 'Location' yang berisi informasi lokasi dalam format "kota, region, negara" dipisahkan menjadi tiga kolom terpisah: 'City', 'Region', dan 'Country'.
# 
# Berikut adalah langkah-langkah yang dilakukan:
# 
# 1. Memisahkan kolom Location: Menggunakan fungsi str.split() untuk memisahkan nilai di kolom 'Location' berdasarkan koma, dan hasilnya disimpan dalam variabel split_location.
# 2. Mengisi nilai kosong: Jika ada baris yang hanya memiliki dua elemen (misalnya hanya ada kota dan negara), maka elemen ketiga (Region) diisi dengan string kosong.
# 3. Menamai kolom baru: Memberikan nama kolom 'City', 'Region', dan 'Country' berdasarkan hasil pemisahan.
# 4. Menghapus spasi ekstra: Menggunakan fungsi str.strip() untuk menghapus spasi tambahan di awal dan akhir setiap elemen di kolom baru.
# 5. Menghapus kolom Location: Menghapus kolom 'Location' yang sudah tidak diperlukan lagi.
# 
# Hasilnya adalah dataframe dengan kolom terpisah untuk 'City', 'Region', dan 'Country', mempermudah analisis lebih lanjut terhadap lokasi pengguna.

# %%
users.isnull().sum()

# %%
empty_string_country = users[users.Country == ''].Country.count()
nan_country = users.Country.isnull().sum()
print(f'There are {empty_string_country} entries with empty strings, and {nan_country} NaN entries in the Country field')

# %% [markdown]
# Kode di atas digunakan untuk menghitung jumlah entri dengan nilai kosong atau tidak terisi pada kolom 'Country' dalam dataframe users.
# 
# 1. Menghitung entri dengan string kosong: Baris pertama (empty_string_country) menghitung jumlah entri di kolom 'Country' yang memiliki string kosong ('').
# 2. Menghitung entri dengan nilai NaN: Baris kedua (nan_country) menghitung jumlah entri di kolom 'Country' yang memiliki nilai NaN (Not a Number), yang menunjukkan nilai yang hilang atau tidak terisi.
# 3. Menampilkan hasil: Hasil kedua perhitungan tersebut dicetak dalam format yang menunjukkan jumlah entri kosong dan NaN di kolom 'Country'.

# %%
import numpy as np
users.Country.replace('', np.nan, inplace=True)

# %% [markdown]
# Kode di atas menggantikan nilai string kosong ('') di kolom 'Country' dalam dataframe users dengan nilai NaN (Not a Number) menggunakan fungsi replace() dari pandas. Hal ini dilakukan untuk menyamakan cara penanganan nilai yang hilang, sehingga entri dengan string kosong dapat dianggap sebagai nilai yang hilang (NaN) untuk analisis lebih lanjut. inplace=True memastikan perubahan dilakukan langsung pada dataframe users tanpa perlu membuat salinan baru.

# %%
users.isnull().sum()

# %%
# Mengisi nilai NaN pada kolom 'Country' dengan 'Other'
users['Country'] = users['Country'].fillna('Other')

# %% [markdown]
# Kode di atas mengisi nilai NaN yang ada pada kolom 'Country' dalam dataframe users dengan string 'Other'.

# %%
# Mengganti nilai 'n/a' pada kolom 'Region' dengan 'Other'
users['Region'] = users['Region'].replace('na', 'Other')

# %% [markdown]
# Kode di atas mengisi nilai na yang ada pada kolom 'Region' dalam dataframe users dengan string 'Other'.

# %%
users.isnull().sum()

# %%
users["Country"].unique().tolist()

# %% [markdown]
# Saya juga mengamati bahwa ada kata-kata "england" dan "united kingdom" yang sebenarnya merujuk pada hal yang sama, jadi saya hanya mempertahankan "england" dengan mengganti "United Kingdom". Saya melakukan hal yang sama untuk "united states of america" dan "usa", serta "l'italia" dan "italy".

# %%
users = users.replace(to_replace =["england, united kingdom","united kingdom"], value = "england",  regex=True)
users = users.replace(to_replace ="united states of america", value = "usa",  regex=True)
users = users.replace(to_replace ="l`italia", value = "italy",  regex=True)
users.Country.value_counts()

# %%
import re

# Fungsi untuk menghapus semua karakter khusus
def remove_special_characters(value):
    if isinstance(value, str):
        # Hanya menyisakan huruf, angka, dan spasi
        return re.sub(r'[^A-Za-z0-9\s]', '', value).strip()
    return value  # Jika bukan string, kembalikan nilai aslinya

# Terapkan pembersihan ke seluruh DataFrame
users = users.applymap(remove_special_characters)

# %% [markdown]
# Sintaks di atas mendefinisikan fungsi remove_special_characters yang menghapus semua karakter khusus (seperti tanda baca atau simbol) dari kolom yang berisi teks. Fungsi ini hanya mempertahankan huruf, angka, dan spasi. Fungsi ini diterapkan ke seluruh DataFrame users menggunakan applymap, sehingga setiap elemen dalam DataFrame yang merupakan string akan dibersihkan dari karakter-karakter khusus. Jika elemen tersebut bukan string, nilai asli akan dipertahankan.

# %%
users.Country.value_counts()

# %%
# Jumlah Pengguna per Negara
country_counts = users['Country'].value_counts().head(10)

# Plot jumlah pengguna per negara
plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar', color='orange')
plt.title('Jumlah Pengguna per Negara', fontsize=16)
plt.xlabel('Negara', fontsize=12)
plt.ylabel('Jumlah Pengguna', fontsize=12)
plt.xticks(rotation=45)

# Adding labels to the bars
for i, v in enumerate(country_counts):
    plt.text(i, v + 100, str(v), ha='center', va='bottom', fontsize=10)

plt.show()

# %%
# Jumlah Pengguna per Kota
city_counts = users['City'].value_counts().head(10)

# Plot jumlah pengguna per kota
plt.figure(figsize=(12, 6))
city_counts.plot(kind='bar', color='green')  # Menampilkan 20 kota teratas
plt.title('Jumlah Pengguna per Kota', fontsize=16)
plt.xlabel('Kota', fontsize=12)
plt.ylabel('Jumlah Pengguna', fontsize=12)
plt.xticks(rotation=45)

# Adding labels to the bars
for i, v in enumerate(city_counts):
    plt.text(i, v + 100, str(v), ha='center', va='bottom', fontsize=10)

plt.show()

# %%
# Jumlah Pengguna per Wilayah (State/Region)
region_counts = users['Region'].value_counts().head(10)

# Plot jumlah pengguna per wilayah
plt.figure(figsize=(12, 6))
region_counts.plot(kind='bar', color='purple')  # Menampilkan 20 wilayah teratas
plt.title('Jumlah Pengguna per Region', fontsize=16)
plt.xlabel('Wilayah', fontsize=12)
plt.ylabel('Jumlah Pengguna', fontsize=12)
plt.xticks(rotation=45)

# Adding labels to the bars
for i, v in enumerate(region_counts):
    plt.text(i, v + 100, str(v), ha='center', va='bottom', fontsize=10)

plt.show()

# %% [markdown]
# ## Popular Based Recommender System

# %% [markdown]
# ### Merge data

# %% [markdown]
# Menggabungkan tabel books, users, dan ratings dalam Satu Tabel

# %%
# Joining books and user ratings into one table 
dataset = ratings.merge(books, on = 'ISBN')
dataset = dataset.merge(users, on = 'User-ID')
dataset = dataset[dataset['Book-Rating'] != 0]
dataset

# %% [markdown]
# ## Popularity Based Recommender System

# %%
popular_df = dataset.groupby('Book-Title').agg(num_rating=('Book-Rating', 'count'),
                                                         avg_rating= ('Book-Rating','mean'))
popular_df = popular_df.reset_index()
popular_df

# %% [markdown]
# Sintaks di atas digunakan untuk membuat DataFrame baru bernama popular_df yang berisi informasi tentang buku yang paling populer. Prosesnya dilakukan dengan cara:
# 
# 1. Mengelompokkan data berdasarkan 'Book-Title': Setiap buku dikelompokkan berdasarkan judul buku.
# 2. Menghitung jumlah rating ('num_rating'): Untuk setiap buku, dihitung berapa banyak rating yang diberikan.
# 3. Menghitung rata-rata rating ('avg_rating'): Untuk setiap buku, dihitung nilai rata-rata dari semua rating yang diberikan.
# 4. Reset Index: Mengembalikan indeks DataFrame agar menjadi urutan standar setelah agregasi.
# 
# Hasil akhir adalah sebuah DataFrame dengan dua kolom utama: jumlah rating (num_rating) dan rata-rata rating (avg_rating) untuk setiap buku.

# %%
popular_df.sort_values('num_rating',ascending=False)

# %%
popular_df.describe()

# %% [markdown]
# Selanjutnya memfilter dan mengurutkan buku berdasarkan popularitas. Pertama, DataFrame popular_df disaring hanya untuk buku yang memiliki lebih dari 200 rating (num_rating > 200). Kemudian, buku-buku tersebut diurutkan berdasarkan nilai rata-rata rating (avg_rating) secara menurun (dari yang tertinggi). Dengan cara ini, hanya buku yang telah banyak dibaca dan memiliki rating tinggi yang akan ditampilkan, sehingga memberikan daftar buku yang paling populer berdasarkan rating dan jumlah pembaca.

# %%
# Popularity is based on the no of people read the book  ('num_raitng' > 200)
# It is based on the rating it got. 
popular_df = popular_df[popular_df['num_rating']>200].sort_values('avg_rating', ascending=False)
popular_df

# %%
popular_df = popular_df.head(20)

# %%
# For the model deployment I need Book-title, Author, Image URL 
popular_df = popular_df.merge(books, on = 'Book-Title').drop_duplicates('Book-Title')[['Book-Title', 
                                                                                       'Book-Author',
                                                                                       'num_rating',
                                                                                      'avg_rating']]

# %% [markdown]
# Berikut merupakan 20 rekomendasi buku berdasarkan popularitas yang diurutkan berdasarkan nilai rata-rata rating (avg_rating).

# %%
popular_df

# %% [markdown]
# ## Collaborative Filtering (Item Based Filtering)

# %%
dataset

# %%
from sklearn.metrics.pairwise import cosine_similarity

# Langkah 1: Filter pengguna dan buku terkenal
user_threshold = 50  # Ambang jumlah buku yang dibaca user
book_threshold = 20  # Ambang jumlah rating yang diterima buku

# Filter pengguna dengan interaksi lebih dari ambang batas
user_counts = dataset.groupby("User-ID").count()["Book-Title"]
active_users = user_counts[user_counts > user_threshold].index
filtered_dataset = dataset[dataset["User-ID"].isin(active_users)]

# Filter buku yang memiliki jumlah rating lebih dari ambang batas
book_counts = filtered_dataset.groupby("Book-Title").count()["Book-Rating"]
popular_books = book_counts[book_counts > book_threshold].index
final_dataset = filtered_dataset[filtered_dataset["Book-Title"].isin(popular_books)]
final_dataset

# %% [markdown]
# Sintaks ini melakukan penyaringan dataset untuk keperluan Collaborative Filtering dengan Item Based Filtering. Pertama, pengguna yang memberikan rating pada lebih dari 50 buku disaring untuk memastikan hanya pengguna aktif yang dipertimbangkan. Kemudian, buku yang memiliki lebih dari 20 rating juga disaring, sehingga hanya buku yang populer yang digunakan. Hasil akhirnya adalah dataset yang hanya mencakup interaksi antara pengguna aktif dan buku populer, yang siap digunakan untuk analisis lebih lanjut atau untuk membuat rekomendasi berbasis kesamaan antara pengguna dan buku.

# %%
# Langkah 2: Buat Pivot Table
pt = final_dataset.pivot_table(index="Book-Title", columns="User-ID", values="Book-Rating")
pt.fillna(0, inplace=True)
pt

# %% [markdown]
# Tujuan dari pembuatan pivot table ini adalah untuk menyusun data dalam format yang memudahkan analisis rekomendasi buku. Dengan mengonversi data menjadi tabel yang menghubungkan judul buku dengan pengguna serta rating yang diberikan, saya dapat menganalisis pola preferensi pengguna terhadap berbagai buku. Ini memudahkan perhitungan similarity (kesamaan) antar buku untuk sistem rekomendasi berbasis collaborative filtering, seperti menghitung cosine similarity untuk memberikan rekomendasi buku kepada pengguna berdasarkan buku yang telah mereka beri rating.

# %%
# Langkah 3: Hitung Cosine Similarity
similarity_score = cosine_similarity(pt)
similarity_score

# %% [markdown]
# Sintaks diaatas menghitung cosine similarity antara berbagai buku berdasarkan rating yang diberikan oleh pengguna.

# %%
# Langkah 4: Fungsi untuk merekomendasikan buku
def recommend_books(book_name, pt, similarity_score, n=10):
    if book_name not in pt.index:
        return "Buku tidak ditemukan dalam data."
    
    index = np.where(pt.index == book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:n+1]

    recommendations = []
    for i in similar_books:
        recommendations.append(pt.index[i[0]])
    
    return recommendations

# %% [markdown]
# Sintaks di atas mendefinisikan fungsi recommend_books yang bertujuan untuk memberikan rekomendasi buku berdasarkan buku yang dimasukkan oleh pengguna.

# %%
# Contoh penggunaan
book_name = "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"
recommendations = recommend_books(book_name, pt, similarity_score, n=10)

print(f"Rekomendasi untuk buku '{book_name}':")
for idx, rec in enumerate(recommendations, 1):
    print(f"{idx}. {rec}")

# %% [markdown]
# Sintaks di atas adalah contoh penggunaan dari fungsi recommend_books yang telah dibuat sebelumnya dengan memberikan 10 rekomendasi buku setelah membaca Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))

# %% [markdown]
# ## Content Based Filtering

# %% [markdown]
# ### Berdasarkan Popularitas Judul Buku

# %% [markdown]
# Pada langkah pertama Content-Based Filtering ini, tujuan utamanya adalah memfilter buku-buku yang populer berdasarkan jumlah rating yang diterima. Pertama, dihitung jumlah rating untuk setiap buku menggunakan groupby dan fungsi transform('count') pada kolom Book-Rating, yang menghitung berapa banyak rating yang diterima setiap buku berdasarkan ISBN-nya. Setelah itu, buku yang memiliki jumlah rating lebih besar atau sama dengan ambang batas yang telah ditentukan (misalnya 80 rating) dipilih sebagai buku populer. Buku-buku ini kemudian disimpan dalam variabel popular_books dan indeksnya di-reset untuk mempermudah analisis lebih lanjut. Langkah ini memastikan hanya buku yang benar-benar populer yang dipertimbangkan dalam rekomendasi berdasarkan rating pengguna.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Langkah 1: Filter buku populer berdasarkan rating (buku yang memiliki banyak rating)
popularity_threshold = 80  # Tentukan ambang batas jumlah rating untuk buku populer
dataset['Total-Ratings'] = dataset.groupby('ISBN')['Book-Rating'].transform('count')  # Menghitung jumlah rating per buku
popular_books = dataset[dataset['Total-Ratings'] >= popularity_threshold]  # Memfilter buku populer
popular_books = popular_books.reset_index(drop=True)
popular_books

# %% [markdown]
# Pada langkah kedua, digunakan TfidfVectorizer untuk membuat matriks TF-IDF berdasarkan judul buku (Book-Title). TfidfVectorizer mengubah teks (judul buku) menjadi representasi numerik yang dapat digunakan untuk analisis lebih lanjut, dengan memberikan bobot pada kata-kata berdasarkan seberapa sering kata tersebut muncul dalam satu dokumen (judul buku) dan seberapa sering kata tersebut muncul di seluruh korpus data. Parameter ngram_range=(1, 2) memungkinkan ekstraksi unigram (kata tunggal) dan bigram (kombinasi dua kata). min_df=1 berarti setiap kata yang muncul setidaknya satu kali akan dipertimbangkan, dan stop_words='english' mengabaikan kata-kata umum dalam bahasa Inggris (seperti "the", "and", dll.). Hasilnya adalah matriks TF-IDF dengan dimensi yang dapat diakses menggunakan shape, yang menunjukkan jumlah fitur (kata unik dan pasangan kata) dan jumlah buku yang diproses.

# %%
# Langkah 2: Membuat TF-IDF Matrix berdasarkan Book-Title
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tf.fit_transform(popular_books['Book-Title'])
tfidf_matrix.shape

# %% [markdown]
# Pada langkah ketiga, dilakukan perhitungan cosine similarity antar buku menggunakan matriks TF-IDF yang telah dibuat sebelumnya. Pertama, matriks TF-IDF dinormalisasi dengan mengubah tipe data matriks menjadi np.float32 untuk memastikan komputasi yang lebih efisien. Kemudian, cosine_similarity digunakan untuk menghitung kesamaan antara buku-buku berdasarkan vektor fitur TF-IDF mereka. Hasilnya adalah matriks yang berisi nilai-nilai cosine similarity antara setiap pasangan buku, di mana nilai mendekati 1 menunjukkan kesamaan yang tinggi dan nilai mendekati 0 menunjukkan kesamaan yang rendah. Matriks ini menggambarkan seberapa mirip judul buku satu dengan yang lainnya berdasarkan konten teks mereka.

# %%
# Langkah 3: Menghitung Cosine Similarity antar Buku
normalized_df = tfidf_matrix.astype(np.float32)  # Normalisasi matrix
cosine_similarities = cosine_similarity(normalized_df, normalized_df)  # Menghitung cosine similarity
cosine_similarities

# %% [markdown]
# Fungsi recommend_books ini digunakan untuk memberikan rekomendasi buku berdasarkan judul buku yang dipilih oleh pengguna. Langkah-langkahnya adalah sebagai berikut:
# 
# 1. Mencari ISBN Buku: Fungsi ini mencari ISBN buku yang dipilih dalam dataset. Jika buku tidak ditemukan, maka fungsi akan mencetak pesan error dan keluar.
# 2. Menemukan Indeks Buku: Indeks buku yang dipilih diambil berdasarkan ISBN.
# 3. Mencari Buku Mirip: Berdasarkan indeks buku, fungsi ini mencari buku-buku yang paling mirip menggunakan nilai cosine similarity yang telah dihitung sebelumnya. Buku-buku yang memiliki nilai cosine similarity tertinggi akan dipilih.
# 4. Menampilkan Rekomendasi: Buku yang memiliki kesamaan tertinggi akan ditampilkan sebagai rekomendasi (dengan jumlah yang ditentukan dalam parameter number).
# 
# Fungsi ini memberikan rekomendasi buku berdasarkan konten yang mirip, seperti yang tercermin dari judul buku yang dipilih.

# %%
# Fungsi untuk merekomendasikan buku berdasarkan nama buku yang dipilih
def recommend_books(bookName, number=10):
    # Cari ISBN dari buku yang dipilih
    try:
        isbn = popular_books.loc[popular_books['Book-Title'] == bookName].reset_index(drop=True).iloc[0]['ISBN']
    except IndexError:
        print(f"Buku '{bookName}' tidak ditemukan dalam dataset.")
        return
    
    content = []
    # Temukan indeks buku yang dipilih
    idx = popular_books.index[popular_books['ISBN'] == isbn].tolist()[0]
    
    # Temukan buku-buku yang paling mirip berdasarkan cosine similarity
    similar_indices = cosine_similarities[idx].argsort()[::-1]  # Urutkan berdasarkan similarity tertinggi
    
    similar_items = []
    for i in similar_indices:
        if popular_books['Book-Title'][i] != bookName and popular_books['Book-Title'][i] not in similar_items and len(similar_items) < number:
            similar_items.append(popular_books['Book-Title'][i])
            content.append(popular_books['Book-Title'][i])

    # Menampilkan buku-buku yang direkomendasikan
    print(f"Recommended Books based on '{bookName}':\n")
    for book in similar_items:
        print(book)

# Contoh: Rekomendasi berdasarkan buku tertentu
bookName = "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"  # Ganti dengan judul buku yang dipilih
recommend_books(bookName)

# %% [markdown]
# ### Berdasarkan Rata-rata Rating Buku

# %% [markdown]
# Langkah pertama ini berfokus pada pembuatan DataFrame yang mengelompokkan buku berdasarkan judul (Book-Title). Kemudian, dihitung dua hal penting:
# 
# 1. num_rating: Jumlah rating yang diberikan untuk setiap buku, menggunakan fungsi count pada kolom Book-Rating.
# 2. avg_rating: Rata-rata rating untuk setiap buku, menggunakan fungsi mean pada kolom Book-Rating.
# 
# Data ini kemudian direset indeksnya untuk memudahkan akses, dan hasil akhirnya adalah DataFrame yang berisi judul buku, jumlah rating yang diterima, dan rata-rata rating untuk setiap buku. Data ini dapat digunakan untuk mengidentifikasi buku-buku populer atau berkinerja baik berdasarkan rating yang diterima.

# %%
# Langkah 1: Filter Buku Berdasarkan Popularitas dan Rata-rata Rating
popular_df = dataset.groupby('Book-Title').agg(num_rating=('Book-Rating', 'count'),
                                               avg_rating=('Book-Rating', 'mean'))
popular_df = popular_df.reset_index()
popular_df

# %% [markdown]
# Pada langkah ini, pertama dilakukan filter pada popular_df untuk hanya memilih buku yang memiliki lebih dari 50 rating. Kemudian, DataFrame diurutkan berdasarkan avg_rating (rata-rata rating) secara menurun agar buku dengan rating tertinggi berada di atas.
# 
# Setelah itu, popular_df digabungkan dengan dataset asli berdasarkan kolom Book-Title menggunakan pd.merge(), yang memungkinkan untuk mendapatkan detail tambahan dari buku-buku tersebut yang ada pada dataset asli (seperti ISBN, pengarang, dll). Buku yang memiliki judul yang sama akan dihapus duplikatnya menggunakan drop_duplicates('Book-Title'). Hasilnya adalah DataFrame popular_books yang berisi buku-buku populer dengan lebih dari 50 rating, diurutkan berdasarkan rata-rata rating tertinggi.

# %%
# Hanya pilih buku dengan jumlah rating > 50
popular_df = popular_df[popular_df['num_rating'] > 50]
popular_df = popular_df.sort_values('avg_rating', ascending=False).reset_index(drop=True)

# Gabungkan dengan dataset asli untuk mendapatkan detail tambahan
popular_books = pd.merge(popular_df, dataset, on='Book-Title').drop_duplicates('Book-Title')
popular_books = popular_books.reset_index(drop=True)
popular_books

# %% [markdown]
# Pada langkah ini, dilakukan pembuatan matriks TF-IDF (Term Frequency-Inverse Document Frequency) berdasarkan kolom Book-Title dari DataFrame popular_books. Fungsi TfidfVectorizer digunakan untuk mengonversi teks judul buku menjadi representasi numerik. Parameter yang digunakan antara lain:
# 
# - ngram_range=(1, 2): Menentukan bahwa n-gram yang digunakan adalah unigrams (kata tunggal) dan bigrams (dua kata berturut-turut).
# - min_df=1: Mengizinkan kata yang muncul dalam satu dokumen atau lebih untuk tetap dipertimbangkan.
# - stop_words='english': Menghapus kata-kata umum dalam bahasa Inggris yang tidak memberi informasi (misalnya "the", "and", "is").
# 
# Matriks tfidf_matrix yang dihasilkan menggambarkan relevansi setiap kata dalam judul buku terhadap koleksi buku yang ada. tfidf_matrix.shape menunjukkan ukuran dari matriks tersebut, yaitu jumlah baris (jumlah buku) dan kolom (jumlah fitur/terminologi yang ditemukan).

# %%
# Langkah 2: Membuat TF-IDF Matrix Berdasarkan Judul Buku
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tf.fit_transform(popular_books['Book-Title'])
tfidf_matrix.shape

# %% [markdown]
# Pada langkah ini, dihitung cosine similarity antar buku berdasarkan matriks TF-IDF yang telah dibuat. Fungsi cosine_similarity digunakan untuk mengukur sejauh mana dua buku (dalam hal ini, dua judul buku) memiliki kesamaan dalam konten teksnya, berdasarkan vektor representasi yang dihasilkan dari matriks TF-IDF.
# 
# Cosine similarity mengukur sudut antara dua vektor dalam ruang vektor, dengan nilai berkisar antara 0 hingga 1, di mana 1 menunjukkan kesamaan sempurna dan 0 menunjukkan tidak ada kesamaan. Matriks cosine_similarities yang dihasilkan berisi nilai-nilai similarity antar setiap pasangan buku, di mana setiap elemen (i, j) dalam matriks menunjukkan tingkat kesamaan antara buku i dan buku j berdasarkan judulnya.

# %%
# Langkah 3: Menghitung Cosine Similarity Antar Buku
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_similarities

# %%
# Langkah 4: Fungsi Rekomendasi Buku
def recommend_books(bookName, number=10):
    try:
        # Cari indeks buku yang dipilih
        idx = popular_books.index[popular_books['Book-Title'] == bookName].tolist()[0]
    except IndexError:
        print(f"Buku '{bookName}' tidak ditemukan dalam daftar buku populer.")
        return pd.DataFrame(columns=["Book-Title", "Book-Author", "avg_rating"])
    
    # Cari buku-buku yang paling mirip berdasarkan cosine similarity
    similar_indices = cosine_similarities[idx].argsort()[::-1]
    
    # Simpan hasil rekomendasi dalam list
    similar_items = []
    for i in similar_indices:
        if popular_books['Book-Title'].iloc[i] != bookName and \
           popular_books['Book-Title'].iloc[i] not in similar_items and \
           len(similar_items) < number:
            similar_items.append(i)

    # Buat dataframe hasil rekomendasi
    recommendations = popular_books.iloc[similar_items][["Book-Title", "Book-Author", "avg_rating"]]
    
    # Urutkan hasil berdasarkan avg_rating secara menurun
    recommendations = recommendations.sort_values(by='avg_rating', ascending=False).reset_index(drop=True)
    
    return recommendations

# Contoh: Rekomendasi berdasarkan judul buku
bookName = "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"  # Ganti dengan judul buku yang dipilih
recommendations = recommend_books(bookName)
recommendations

# %% [markdown]
# Fungsi recommend_books di atas digunakan untuk memberikan rekomendasi buku berdasarkan judul buku yang dipilih. Prosesnya dimulai dengan mencari indeks buku yang dipilih dalam daftar buku populer. Setelah itu, dihitung buku-buku yang paling mirip menggunakan nilai cosine similarity yang telah dihitung sebelumnya. Buku-buku yang mirip dengan buku yang dipilih akan disortir berdasarkan rating rata-rata (avg_rating) secara menurun, dan hanya buku-buku dengan rating terbaik yang akan dimasukkan dalam rekomendasi. Fungsi ini mengembalikan dataframe yang berisi daftar buku yang direkomendasikan, termasuk judul buku, pengarang, dan rating rata-rata, dengan batasan jumlah rekomendasi yang diinginkan.

# %% [markdown]
# 


