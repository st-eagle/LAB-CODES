import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hdbscan
import cv2
from sklearn.manifold import TSNE

# Step 1: Load and Explore the Dataset
# Load the LFW (Labeled Faces in the Wild) dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Dataset information
print(f"Dataset shape: {lfw_people.images.shape}")
print(f"Number of classes (people): {len(lfw_people.target_names)}")
print(f"Feature vector length: {lfw_people.data.shape[1]}")

# Visualize a few images from the dataset
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(lfw_people.images[i], cmap='gray')
    ax.axis('off')
plt.show()

# Step 2: Data Preprocessing
# Flatten the image data and standardize it
X = lfw_people.data  # Flattened images
y = lfw_people.target

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Dimensionality Reduction using PCA
pca = PCA(n_components=50)  # Reduce to 50 principal components
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio (How much information each principal component holds)
print(f"Explained variance ratio of PCA components: {np.cumsum(pca.explained_variance_ratio_)}")

# Step 4: Apply HDBSCAN Clustering
# Perform HDBSCAN clustering on the reduced dataset (PCA components)
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
cluster_labels = clusterer.fit_predict(X_pca)

# Display the clustering results
print(f"Number of clusters: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
print(f"Number of noise points: {list(cluster_labels).count(-1)}")


# Step 5: Introduce Real-Life Challenges

# 5.1: Noisy Data Simulation
def add_noise_to_images(X, noise_factor=0.2):
    noisy_X = X + noise_factor * np.random.randn(*X.shape)
    noisy_X = np.clip(noisy_X, 0, 1)  # Clip to ensure the pixel values remain valid
    return noisy_X


X_noisy = add_noise_to_images(X_scaled)
X_noisy_pca = pca.transform(X_noisy)

# Apply HDBSCAN on noisy data
clusterer_noisy = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
cluster_labels_noisy = clusterer_noisy.fit_predict(X_noisy_pca)


# Step 6: Visualize and Analyze Results
# Create scatter plot of the PCA-reduced data and HDBSCAN clustering results
def plot_clustered_data(X_reduced, cluster_labels, title="Clustered Data"):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='Spectral', s=50, edgecolor='k')
    plt.title(title)
    plt.colorbar()
    plt.show()


# Plot the clustering result of original data
plot_clustered_data(X_pca, cluster_labels, "Clustering Result (Original Data)")

# Plot the clustering result of noisy data
plot_clustered_data(X_noisy_pca, cluster_labels_noisy, "Clustering Result (Noisy Data)")

# Step 7: Real-Life Applications
# 7.1: Test on New Data (Use a random image from the dataset)
new_image = X[100].reshape(1, -1)  # Take a random image for testing
new_image_scaled = scaler.transform(new_image)
new_image_pca = pca.transform(new_image_scaled)
new_image_cluster = clusterer.predict(new_image_pca)
print(f"Cluster label for new image: {new_image_cluster[0]}")

# 7.2: Identify Representative Images for Clusters
# Find the representative image closest to the center of each cluster
cluster_centers = []
for cluster in set(cluster_labels):
    if cluster != -1:  # Skip noise points
        cluster_data = X_pca[cluster_labels == cluster]
        cluster_center = cluster_data.mean(axis=0)
        cluster_centers.append(cluster_center)

# Find the closest image to each cluster center
representative_images = []
for center in cluster_centers:
    distances = np.linalg.norm(X_pca - center, axis=1)
    closest_image_idx = np.argmin(distances)
    representative_images.append(closest_image_idx)

# Visualize the representative images
fig, axes = plt.subplots(1, len(representative_images), figsize=(15, 5))
for ax, idx in zip(axes, representative_images):
    ax.imshow(lfw_people.images[idx], cmap='gray')
    ax.axis('off')
plt.show()

# Step 8:. Evaluate Clustering Quality
# Calculate Silhouette Score
sil_score = silhouette_score(X_pca, cluster_labels)
print(f"Silhouette Score {sil_score}")
