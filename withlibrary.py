import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Data penderita hipertensi
data_hipertensi = {
    'kolesterol': [240, 260, 250, 245, 255, 265, 270, 275, 260, 250, 245, 260],
    'asam_urat': [7.0, 7.5, 7.2, 7.1, 7.3, 7.4, 7.6, 7.8, 7.5, 7.2, 7.1, 7.3],
    'label': [1]*12  # 1 menandakan hipertensi
}

# Data orang normal
data_normal = {
    'kolesterol': [180, 190, 200, 175, 185, 195, 200, 205, 190, 180, 175, 195],
    'asam_urat': [5.0, 5.5, 5.2, 5.1, 5.3, 5.4, 5.6, 5.8, 5.5, 5.2, 5.1, 5.3],
    'label': [0]*12  # 0 menandakan tidak hipertensi
}

# Gabungkan data menjadi satu DataFrame
data_hipertensi_df = pd.DataFrame(data_hipertensi)
data_normal_df = pd.DataFrame(data_normal)
data = pd.concat([data_hipertensi_df, data_normal_df], ignore_index=True)

# Pisahkan fitur dan label
X = data[['kolesterol', 'asam_urat']]
y = data['label']

# Bagi data menjadi data latih dan data tes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Buat dan latih model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediksi menggunakan data tes
y_pred = knn.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model: {accuracy*100:.2f}%')

# Contoh data baru untuk pengujian dengan nama fitur yang sesuai
data_baru = pd.DataFrame({
    'kolesterol': [250, 185],
    'asam_urat': [7.0, 5.3]
})

# Prediksi menggunakan model KNN
prediksi_baru = knn.predict(data_baru)
print(f'Prediksi untuk data baru: {prediksi_baru}')
