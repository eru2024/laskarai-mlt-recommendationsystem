# Recommendation System for Indonesia Tourism Destination - Fatih El Haq

## Project Overview

Indonesia's tourism sector plays a crucial role in its economy, contributing significantly to foreign exchange revenues and regional development (The Jakarta Post, 2018). However, the abundance of diverse destinations across the archipelago often overwhelms potential visitors, leading to decision fatigue. This challenge underscores the necessity for an effective recommendation system that can assist tourists in selecting destinations that align with their preferences and interests.

Recent studies have demonstrated the effectiveness of various recommendation algorithms in enhancing the tourism experience. For instance, a study on Bali's tourism sector employed a Weighted Hybrid method combining Collaborative Filtering and Content-Based approaches, yielding improved accuracy in destination suggestions (Pratama et al., 2023). Similarly, research focused on Madura Island utilized a Collaborative Filtering-based system with modified Cosine similarity and Convolutional Neural Networks, achieving a low RMSE of 0.2579, indicating precise personalized recommendations (Permana et al., 2024). 
The implementation of such systems is increasingly relevant in the context of e-tourism, which emphasizes the integration of advanced technologies like Artificial Intelligence and Big Data to enhance tourism experiences (Samara et al., 2020). By adopting these technologies, Indonesia can offer tailored travel suggestions that cater to individual preferences, thereby improving tourist satisfaction and supporting sustainable tourism development.

In conclusion, addressing the information overload faced by tourists through personalized recommendation systems is essential for optimizing travel experiences in Indonesia (Pratama et al., 2023). Leveraging advanced algorithms and technologies will not only assist tourists in making informed decisions but also contribute to the sustainable growth of the nation's tourism industry.

## Business Understanding

Indonesia's tourism sector offers a wide range of destinations, from natural wonders and cultural heritage sites to modern entertainment venues. Despite this diversity, many domestic and international tourists face difficulties in identifying destinations that align with their interests, largely due to the overwhelming number of options and the absence of personalized guidance. This situation often leads to inefficient travel planning and underutilization of tourism potential. Fortunately, available datasets include curated information on Indonesian tourism destinations, their categories, and user-generated ratings, offering an opportunity to develop a data-driven recommendation system.

### Problem Statements
To address the challenges in personalized destination discovery, this project defines the following problem statements:
- How can a personalized destination recommendation system be created using content-based filtering techniques based on destination data?
- How can other potentially preferred and unvisited destinations be recommended to users based on existing ratings data?

### Goals
To answer the problem statements, the project aims to achieve the following goals:
- Develop a personalized recommendation system that suggests destinations similar to user preferences using content-based filtering.
- Recommend destinations that match user preferences and have not yet been visited by leveraging collaborative filtering techniques.

### Solution Statements
To achieve these goals, the project applies two recommendation system approaches:
- **Content-Based Filtering**: Utilizes TF-IDF Vectorizer to transform destination text attributes into weighted numerical vectors and applies Cosine Similarity to calculate the similarity between these vectors, enabling the system to recommend destinations similar to those preferred by the user.
- **Collaborative Filtering**: Implements RecommenderNet, a deep learning model that captures latent features from user-place interaction data (ratings) to predict preferences for unrated destinations, thereby supporting personalized recommendations.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

## References
Permana, K. E., Rahmat, A. B., Wicaksana, D. A., & Ardianto, D. (2024). Collaborative filtering-based Madura Island tourism recommendation system using RecommenderNet. BIO Web of Conferences, 146, 01080. https://doi.org/10.1051/bioconf/202414601080

Pratama, D. E., Nurjanah, D., & Nurrahmi, H. (2023). Tourism Recommendation System using Weighted Hybrid Method in Bali Island. JURNAL MEDIA INFORMATIKA BUDIDARMA, 7(3), 1189. https://doi.org/10.30865/mib.v7i3.6409

Samara, D., Magnisalis, I., & Peristeras, V. (2020). Artificial intelligence and big data in tourism: A systematic literature review. Journal of Hospitality and Tourism Technology, 11(2), 343–367. https://doi.org/10.1108/jhtt-12-2018-0118

The Jakarta Post. (2018, October 23). Indonesian tourism set to beat Thailand in 5 years. The Jakarta Post. https://www.thejakartapost.com/news/2018/10/23/indonesian-tourism-set-to-beat-thailand-in-5-years.html



**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
