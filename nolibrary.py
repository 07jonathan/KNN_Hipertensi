import numpy as np
import pandas as pd
from collections import Counter

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

# Fungsi untuk melakukan prediksi KNN
def knn_predict(train, test_row, num_neighbors):
    distances = []
    for index, train_row in train.iterrows():
        dist = euclidean_distance(np.array(train_row[:-1]), np.array(test_row))
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    output_values = [row[-1] for row in neighbors]
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction

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

# Pisahkan data latih dan data tes secara manual (misalnya, 70% data latih dan 30% data tes)
train = data.sample(frac=0.7, random_state=42)
test = data.drop(train.index)

# Prediksi untuk data tes
num_neighbors = 3
predictions = []
for index, test_row in test.iterrows():
    prediction = knn_predict(train, np.array(test_row[:-1]), num_neighbors)
    predictions.append(prediction)

# Evaluasi model
accuracy = np.sum(np.array(predictions) == np.array(test['label'])) / len(test)
print(f'Akurasi model: {accuracy*100:.2f}%')

# Contoh data baru untuk pengujian
data_baru = pd.DataFrame({
    'kolesterol': [250, 185],
    'asam_urat': [7.0, 5.3]
})

# Prediksi menggunakan model KNN
prediksi_baru = []
for index, row in data_baru.iterrows():
    prediksi_baru.append(knn_predict(train, np.array(row), num_neighbors))

print(f'Prediksi untuk data baru: {prediksi_baru}')
