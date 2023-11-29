# Laporan Proyek Machine Learning - Roissyah Fernanda Khoiroh

## Domain Proyek
Biaya layanan kesehatan yang meningkat pesat telah menjadi tantangan besar di beberapa negara, bukti yang ada menunjukkan bahwa biaya perawatan kesehatan telah terakumulasi pada sejumlah besar BMI [1]. Faktor-faktor demografi pasien seperti usia, jenis kelamin, merokok atau tidak, wilayah, tanggungan anak, dan BMI dapat memengaruhi biaya tanggungan asuransi kesehatan. Oleh karena itu, diperlukan model yang dapat memprediksi biaya tersebut berdasarkan faktor-faktor demografi pasien menjadi penting untuk dikembangkan. Dengan mengetahui prediksi biaya pengobatan dapat membantu agen asuransi untuk menghitung premi asuransi yang lebih akurat, sehingga dapat meningkatkan kepuasan pelanggan dan mengurangi risiko kerugian. Prediksi juga membantu rumah sakit untuk memperkirakan kebutuhan dana dan sumber daya untuk perawatan pasien, sehingga dapat meningkatkan efisiensi dan kualitas layanan.

Penelitian sebelumnya bertujuan untuk memprediksi pasien mana yang menghabiskan biaya pengobatan besar di tahun berikutnya dengan membandingkan algortima random forest (RF), gradient boosting machine (GBM), artificial neural network (ANN) dan logistic regression (LR). Adapun data yang digunakan berupa klaim dana sakit dan biaya yang dikeluarkan secara rutin pada tahun 2016, 2017, dan 2018 [2]. Penelitian lainnya berupaya ntuk membangun model regresi linear guna memprediksi berapa besar biaya pengobatan yang harus dikeluarkan oleh asuransi jika diketahui variabel-variabel seperti jenis kelamin, BMI, merokok atau tidak, usia, serta jumlah anak yang ditanggung. Peneliti menyebutkan penting untuk memprediksi biaya pengobatan tersebut karena membantu untuk menciptakan strategi pencegahan obesitas yang hemat biaya, sebagaimana dalam penelitian terpisah lainnya disebutkan BMI, penuaan, merokok, dan faktor lainnya semuanya terkait dengan biaya perawatan medis pribadi yang lebih besar [1]. 

## Business Understanding
Biaya pengobatan yang dikeluarkan oleh asuransi kesehatan merupakan salah satu faktor yang penting untuk dipertimbangkan dalam menentukan premi yang harus dibayarkan oleh nasabah. Oleh karena itu, penting untuk memiliki model yang akurat dalam memprediksi besar biaya pengobatan tersebut. Model regresi yang dikembangkan dalam proyek ini akan menggunakan 6 variabel prediktor, yaitu: jenis kelamin, BMI, merokok atau tidak, usia, jumlah anak yang ditanggung, wilayah asal.

Variabel-variabel tersebut dipilih karena diyakini memiliki hubungan yang signifikan dengan biaya pengobatan. Misalnya, jenis kelamin, BMI, dan merokok diketahui dapat meningkatkan risiko penyakit tertentu yang dapat menyebabkan biaya pengobatan yang lebih tinggi. Usia juga merupakan faktor yang penting, karena biaya pengobatan umumnya meningkat seiring dengan bertambahnya usia. Jumlah anak yang ditanggung juga dapat mempengaruhi biaya pengobatan, karena anak-anak umumnya lebih rentan terhadap penyakit. Wilayah asal juga dapat mempengaruhi biaya pengobatan, karena biaya layanan kesehatan dapat bervariasi di setiap wilayah. Di samping itu, alasan penggunaan variabel tersebut juga didasarkan oleh penelitian yang dilakukan oleh [1], hal pembeda pada proyek ini terdapat 1 tambahan variabel prediktor yaitu wilayah.

Model regresi yang dikembangkan diharapkan dapat membantu pihak asuransi dalam:
- Merancang produk asuransi yang lebih sesuai dengan kebutuhan pelanggan.
- Memberikan informasi yang lebih akurat kepada pelanggan tentang manfaat dan biaya asuransi.
- Meningkatkan efisiensi proses klaim

Sedangkan untuk rumah sakit, dapat membantu:
- Melakukan perencanaan anggaran dan belanja modal yang lebih akurat.
- Meningkatkan efisiensi penggunaan sumber daya, misalnya dengan mengoptimalkan penggunaan kamar rawat inap.
- Meningkatkan mutu layanan, misalnya dengan meningkatkan kualitas perawatan pasien dengan biaya yang lebih terjangkau.

### Problem Statements
- Variabel apa saja yang berpengaruh besar dalam memprediksi biaya pengobatan?
- Model regresi apa yang bekerja paling baik dalam memprediksi besar biaya pengobatan?

### Goals
- Untuk mengetahui variabel-variabel yang paling berpengaruh besar dalam memprediksi biaya pengobatan
- Untuk mengetahui model regresi terbaik dalam memprediksi besar biaya pengobatan 

### Solution statements
- Menggunakan feature importance untuk mengetahui variabel-variabel yang paling berpengaruh besar dalam memprediksi biaya pengobatan
- Membandingkan performa model Gradient Boosting Regressor, Random Forest Regressor, dan Multiple Linear Regression dalam memprediksi biaya pengobatan
- Melakukan hyperparameter tuning pada model terbaik agar dihasilkan error yang lebih kecil
- Memanfaatkan RMSE sebagai metrik evaluasi untuk mengukur seberapa baik performa model dalam memprediksi biaya pengobatan 

## Data Understanding
Proyek ini menggunakan dataset dari Kaggle yang berjudul [Medical Cost Performance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance?resource=download) dengan jumlah baris 1337 dan fitur yang dimiliki sebagaimana yang tercantum pada Tabel 1 berikut.

Tabel 1. Informasi variabel pada dataset
| **Nama variabel** | **Keterangan** |
| --- | --- |
| age | usia penerima manfaat utama |
| sex | jenis kelamin pemegang polis |
| bmi | indeks massa tubuh, menggunakan rasio tinggi badan terhadap berat badan (kg/m^2), idealnya 18,5 hingga 24,9 |
| children | jumlah tanggungan |
| smoker | perokok atau tidak |
| region | 4 wilayah pemukiman penerima manfaat di AS yaitu wilayah timur laut, tenggara, barat daya, barat laut |
| charges | biaya pengobatan individu yang ditanggung oleh asuransi kesehatan |

Gambar 1. Korelasi antar variabel
![korelasi](https://github.com/roissyahf/dicoding-submission/assets/94748266/9a5b42c8-cd41-4bd7-98ce-0822806d81b9)

Dataset sudah bebas dari nilang kosong dan juga inconsistent value, hanya terdapat 1 nilai duplikat yang kemudian dihilangkan. Setelah melalui tahapan Data Cleaning, dilakukan eksplorasi lebih dalam lagi salah satunya dengan melihat korelasi antar variabel. Berdasarkan Gambar 1, dapat diamati hubungan antar variabel lemah dengan skor korelasi kurang dari 0.3. Variabel charges dan smoker memiliki skor korelasi tertinggi di antara variabel lainnya, yaitu 0.79 yang menandakan adanya hubungan positif kuat antara keduanya. Pemilihan age, sex, bmi, children, dan smoker sebagai variabel prediktor juga mengacu pada penelitian yang dilakukan oleh [1], hal pembedanya pada proyek ini terdapat 1 tambahan variabel prediktor yaitu wilayah.

## Exploratory Data Analysis (EDA)
Gambar 2. Visualisasi bagian I
![bagian_1](https://github.com/roissyahf/dicoding-submission/assets/94748266/d464516d-3e3d-427e-9ecd-6a5ca1e625b3)

Gambar 3. Visualisasi bagian II
![bagian_2](https://github.com/roissyahf/dicoding-submission/assets/94748266/722975c9-9e48-48f9-a080-f93133ce6239)

Berdasarkan visualisasi pada Gambar 2 dan 3, diperoleh temuan:
1. Usia pasien cukup menyebar rata mulai dari 18 hingga lebih dari 60 tahun. Namun pasien dengan usia 18 tahun berjumlah 160 yang membuat distribusi usia pasien menceng ke kanan.
2. Perokok didominasi oleh lelaki, lalu proporsi perokok dan non perokok yaitu 20.5% dibanding 79.5%
3. Sebanyak 573 pasien tidak memiliki tanggungan anak, lalu disusul oleh pasien dengan tanggungan 1 anak, 2 anak, 3 anak, 4 anak, dan terakhir 5 anak
4. Penyebaran tempat tinggal pasien tergolong seimbang khususnya di wilayah southwest, northwest, dan northeast. Sebanyak 364 pasien bermukim di southeast.
5. Untuk pasien dengan BMI < 30 dikenakan biaya tagihan yang mayoritas menyentuh angka < $10000, dengan biaya tertinggi mencapai $38000. Sedangkan untuk pasien dengan BMI > 30, biaya pengobatan menyebar di angka < $16000 dengan tagihan tertinggi mencapai $60000.
6. Terdapat perbedaan biaya antara pasien perokok dan non-perokok. Biaya pengobatan pada pasien perokok menyebar di angka $15000 - $60000 dengan tagihan paling sering mencapai $40000 dan $20000. Sedangkan untuk non perokok, biaya tagihan menyebar mulai dari $1000 hingga $40000, dengan biaya yang paling sering muncul yaitu < $5000.

Temuan tersebut menghasilkan kesimpulan, variabel smoker, dan BMI berpengaruh terhadap besar kecilnya biaya pengobatan pasien.

## Data Preparation
- Sebelum dilakukan pemodelan, variabel kategorikal yaitu sex, smoker, dan region perlu diubah menjadi variabel numerikal dengan menggunakan Label Encoder.
- Lalu dilakukan pemisahan antara X yang merupakan variabel prediktor: age, sex, bmi, children, smoker, region dan y yang merupakan variabel target: charges.

Gambar 4. Distribusi variabel charges sebelum dan sesudah transformasi log
![before_after_log](https://github.com/roissyahf/dicoding-submission/assets/94748266/a8be1a4d-efbc-4a85-b512-6f70955fb1d2)

- Berdasarkan Gambar 4, terlihat variabel charges menceng ke kanan maka perlu dilakukan transformasi logaritma untuk memperbaiki distribusi data. Transformasi logaritma dapat membantu untuk mengurangi dampak nilai-nilai outlier tersebut, sehingga model dapat bekerja lebih baik. 
- Memisahkan data ke dalam training set untuk melatih model, dan test set untuk mengevaluasi model dengan rasio 0.75:0.25

## Model Development
Terdapat 3 model yang akan dibandingkan yaitu Gradient Boosting Regressor, Random Forest Regressor, dan Multiple Linear Regression menggunakan default parameter. Mulanya akan dilakukan cross validation dengan k-fold sebanyak 5 (cv=5), lalu dilakukan model training menggunakan data training. Berikut ringkasan parameter yang digunakan dan rerata RMSE dari hasil cross validation:

Tabel 2. Baseline model beserta parameternya
| **Nama model** | **Parameter** | **Rerata RMSE saat fit time** | **RMSE saat test time** | **MSE saat test time** | **R^2 saat test time** |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting Regressor | learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42 | 0.5379 | 0.4882 | 0.2383 | 0.8726 |
| Random Forest Regressor | n_estimators=100, random_state=42 | 0.5751 | 0.5507 | 0.3032 | 0.2860 | 0.8379 |
| Linear Regression | - | 0.6406 | 0.5757 | 0.3315 | 0.8228 |

Berdasarkan Tabel 2, Gradient Boosting Regressor (GBR) menjadi model terbaik karena memiliki RMSE dan MSE terkecil di antara kedua model lainnya. Di samping itu, GBR juga menghasilkan skor koefisien determinasi (R^2) tertinggi di antara kedua model lainnya, yaitu sebesar 0.8726. GBR mampu bekerja lebih baik daripada Random Forest Regressor (RFR) karena GBR mampu mengoreksi error lalu melakukan perbaikan secara otomatis, GBR juga mampu mengenali pola kompleks dalam data.

Dari 6 variabel prediktor yang digunakan, dengan menggunakan feature importance pada model GBR ditemukan bahwasannya variabel smoker paling berpengaruh terhadap model sengan skor importance sebesar 0.494253, lalu disusul oleh variabel age dengan skor importance 0.387697. Variabel bmi, children, region, sex hanya berkontribusi kecil dalam memprediksi biaya pengobatan sebagaimana yang dapat diamati pada Gambar 5.

Gambar 5. Feature Importance pada model Gradient Boosting Regressor
![importance](https://github.com/roissyahf/dicoding-submission/assets/94748266/3c824725-0af0-41e7-9d34-25ac8f4c091e)

Selanjutnya, dilakukan hyperparameter tuning terhadap model Gradient Boosting Regressor dengan memanfaatkan GridSearchCV untuk mencari parameter terbaik dari serangkaian kombinasi yang dicoba dari dictionary berikut:
```
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}
``` 
Setelah itu, dilakukan training ulang kembali menggunakan parameter terbaik. Diperoleh hasil RMSE mengalami penurunan menjadi 0.4774 dengan menggunakan pengaturan parameter sebagai berikut:
```
Best Hyperparameters: {
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_leaf': 4,
    'min_samples_split': 10,
    'n_estimators': 50,
    'subsample': 0.9}
```

## Evaluation
RMSE (Root Mean Square Error) menjadi metrik utama yang digunakan untuk membandingkan performa model. RMSE menghitung kesalahan prediksi dengan mengkuadratkan selisih nilai aktual (y_test_log) dengan nilai prediksi (y_pred), lalu dibagi dengan banyaknya data pada x_test, dan hasilnya diakarkan. Semakin kecil skor RMSE, semakin baik performa model karena nilai prediksi yang dihasilkan semakin mendekati nilai aktual. Digunakan juga Mean Square Error (MSE) dan koefisien determinasi (R^2) dalam mengevaluasi performa model. Koefisien determinasi memiliki rentang nilai antara 0 hingga 1 yang mengukur kemampuan variabel-variabel independen dalam memprediksi variabel dependen. 

Setelah pemilihan model terbaik dari ketiga model yang diusulkan, diperoleh model Gradient Boosting Regressor (GBR) keluar sebagai model terbaik dengan perolehan skor RMSE dan MSE terkecil di antara kedua model lainnya, koefisien determinasinya juga paling besar sebagaimana yang tercantum pada Tabel 2. Selanjutnya, dilakukan hyperparameter tuning pada model GBR tersebut, hasilnya skor RMSE turun dari yang semula 0.4882 menjadi 0.4774. Koefisien determinasi juga mengalami kenaikan, dari 0.8726 menjadi 0.8782. Dapat diinterpretasikan, model GBR hasil tuning semakin baik dalam memprediksi biaya pengobatan yang dibuktikan dengan penurunan skor RMSE serta kenaikan koefisien determinasi. 

## Conclusion
- Variabel smoker, age, dan bmi berkontribusi besar dalam memprediksi biaya pengobatan.
- Gradient Boosting Regressor menjadi model terbaik dibandingkan Random Forest Regressor dan Linear Regression karena dihasilkan RMSE terkecil yaitu 0.4774 dan skor determinasi terbesar yaitu 0.8726.

Referensi:
[[1]](https://doi.org/10.1155/2022/7969220) Ahmed I. Taloba, Rasha M. Abd El-Aziz, Huda M. Alshanbari, Abdal-Aziz H. El-Bagoury, "Estimation and Prediction of Hospitalization and Medical Care Costs Using Regression in Machine Learning", Journal of Healthcare Engineering, vol. 2022, 2022.

[[2]](https://doi.org/10.1371/journal.pone.0279540) Langenberger B, Schulte T, Groene O. "The application of machine learning to predict high-cost patients: A performance-comparison of different models using healthcare claims data," PLoS ONE, vol. 18(1), 2023.
