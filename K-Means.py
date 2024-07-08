import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Langkah 1: Membaca gambar
image = cv2.imread('image.png')  # ganti 'image.jpg' dengan path gambar Anda
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Langkah 2: Mengubah gambar ke bentuk data yang dapat digunakan oleh K-Means
pixels = image.reshape(-1, 3)

# Langkah 3: Menentukan jumlah cluster dan menginisialisasi K-Means
k = 5  # jumlah cluster
kmeans = KMeans(n_clusters=k, random_state=42)

# Langkah 4: Menjalankan algoritma K-Means
kmeans.fit(pixels)

# Langkah 5: Mendapatkan label cluster dan membuat gambar yang disegmentasi
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
segmented_img = segmented_img.astype('uint8')

# Langkah 6: Menampilkan gambar asli dan gambar yang disegmentasi
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title('Gambar Disegmentasi')
plt.axis('off')

plt.show()
