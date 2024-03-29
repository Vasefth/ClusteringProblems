import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score, adjusted_rand_score
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Reshape

# Load the Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the pixel values of the images to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Split the training data into training and validation sets
# Here, 20% of the training set is used as the validation set
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Normalize the validation images
val_images = val_images / 255.0

results_df = pd.DataFrame(columns=[
    'Dimensionality Reduction Technique',
    'Clustering Algorithm',
    'Dimensionality Reduction Time (s)',
    'Clustering Time (s)',
    'Number of Suggested Clusters',
    'Calinski-Harabasz Index',
    'Davies-Bouldin Index',
    'Silhouette Score',
    'Adjusted Rand Index Score'
])

#This is Step 3 and this is where the loop starts
def apply_pca(data):
  #Flatten the images
  flattened_data = data.reshape((data.shape[0], -1))
  pca = PCA(n_components=20)
  reduced_data = pca.fit_transform(flattened_data)
  return reduced_data, pca

def apply_stacked_autoencoder(data, validation_data):
    # Define the encoder
    encoder = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu')  # Compressed representation
    ])

    # Define the decoder
    decoder = Sequential([
        Dense(64, activation='relu', input_shape=(32,)),
        Dense(128, activation='relu'),
        Dense(784, activation='sigmoid'),
        Reshape((28, 28))
    ])

    # Combine into an autoencoder
    autoencoder = Sequential([encoder, decoder])

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # Train the autoencoder
    autoencoder.fit(data, data, epochs=50, batch_size=256, validation_data=validation_data, callbacks=[early_stopping])

    # Use the encoder to reduce dimensionality
    reduced_data = encoder.predict(data)
    return reduced_data, autoencoder


def apply_lda(data, labels):
  #Flatten the images
  flattened_data = data.reshape((data.shape[0], -1))

  #Initialize LDA
  #The number of components can be set to min(n_classes - 1, n_features)
  lda = LDA(n_components=9)

  #Fit and transform the data
  reduced_data = lda.fit_transform(flattened_data, labels)
  return reduced_data, lda


#Define different dimensionality techniques
dimensionality_reduction_techniques = [
    ("PCA", apply_pca),
    ("Stacked Autoencoder", apply_stacked_autoencoder),
    ("LDA", apply_lda)
]


#The following is for Step 4
def select_random_images_by_class(images, labels, num_classes=10):
  random_images = []
  np.random.seed(42)  # For reproducibility
  for i in range(num_classes):
      class_indices = np.where(labels == i)[0]
      random_index = np.random.choice(class_indices)
      random_images.append(images[random_index])
  return random_images


#The following is for Step 5
def plot_reduced_data(reduced_data, labels, technique_name):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='tab10')
    plt.title(f'Reduced Data Visualization using {technique_name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

#The following is for step 7
def apply_minibatch_kmeans(data, n_clusters=10, n_init=3):
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init)
    mbkmeans.fit(data)
    return mbkmeans.labels_

def apply_dbscan(data, eps=5, min_samples=10):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    return dbscan.labels_

def apply_agglomerative(data, n_clusters=10):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative.fit(data)
    return agglomerative.labels_

#The following is for step 4 also
#Select random images
random_images = select_random_images_by_class(train_images, train_labels)

# Dictionary to store cluster labels
cluster_labels_dict = {}

pca_model = None
lda_model = None
autoencoder_model = None  # Initialize variable to store the autoencoder model

for name, technique in dimensionality_reduction_techniques:
    start_time_dr = time.time()

    if name == "PCA":
        reduced_data, pca_model = apply_pca(train_images)
        plot_reduced_data(reduced_data, train_labels, "PCA")
    elif name == "Stacked Autoencoder":
        reduced_data, autoencoder_model = apply_stacked_autoencoder(train_images, (val_images, val_images))
    else:
       reduced_data, lda_model = apply_lda(train_images, train_labels)
       plot_reduced_data(reduced_data, train_labels, "LDA")

    end_time_dr = time.time()
    dr_time = end_time_dr - start_time_dr
    print(f"{name} completed. Time taken: {dr_time} seconds")

    # Display original and reconstructed images for Stacked Autoencoder
    if name == "Stacked Autoencoder" and autoencoder_model is not None:
        reconstructed_images = autoencoder_model.predict(np.array(random_images))
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i, (original, reconstructed) in enumerate(zip(random_images, reconstructed_images)):
            axes[0, i].imshow(original, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Class {i}\nOriginal", fontsize=10)

            axes[1, i].imshow(reconstructed, cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Reconstructed", fontsize=10)

        plt.tight_layout()
        plt.show()

    #Step 6
    #Flatten the test data for PCA and LDA
    flattened_test_images = test_images.reshape((test_images.shape[0], -1))

    # Apply PCA to test data
    if pca_model is not None:
        encoded_test_pca = pca_model.transform(flattened_test_images)

    # Apply LDA to test data
    if lda_model is not None:
        encoded_test_lda = lda_model.transform(flattened_test_images)

    # Apply Autoencoder to test data
    if autoencoder_model is not None:
        encoded_test_autoencoder = autoencoder_model.predict(test_images)

        # Flatten the output of the autoencoder
        encoded_test_autoencoder_flattened = encoded_test_autoencoder.reshape(encoded_test_autoencoder.shape[0], -1)

        # Reduce dimensionality to 2D for clustering
        #pca_2d = PCA(n_components=2)
        #encoded_test_autoencoder_2d = pca_2d.fit_transform(encoded_test_autoencoder_flattened)

    # Step 7 - Clustering
    # Prepare data for clustering
    if name == "PCA":
        clustering_data = encoded_test_pca
    elif name == "LDA":
        clustering_data = encoded_test_lda
    elif name == "Stacked Autoencoder":
        clustering_data = encoded_test_autoencoder_flattened
    else:
        continue  # Skip if none of the models are applicable

    for cluster_algo, cluster_func in [
        ('MiniBatch KMeans', apply_minibatch_kmeans),
        ('DBSCAN', apply_dbscan),
        ('Agglomerative', apply_agglomerative)
    ]:
        # Measure the clustering time
        start_time_cluster = time.time()
        labels = cluster_func(clustering_data)
        end_time_cluster = time.time()
        cluster_time = end_time_cluster - start_time_cluster

        # Save the labels in the dictionary
        key = f"{name}_{cluster_algo}"
        cluster_labels_dict[key] = labels

        # Plot diagram
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=clustering_data[:, 0], y=clustering_data[:, 1], hue=labels, palette='tab10')
        plt.title(f'{cluster_algo} Clustering with {name}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()

        # Calculate the number of clusters (excluding noise for DBSCAN)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculate metrics
        ch_index = calinski_harabasz_score(clustering_data, labels)
        db_index = davies_bouldin_score(clustering_data, labels)
        silhouette_avg = silhouette_score(clustering_data, labels)
        ari_score = adjusted_rand_score(test_labels, labels)

        # Create a new DataFrame for the current iteration's results
        new_row = pd.DataFrame({
            'Dimensionality Reduction Technique': [name],
            'Clustering Algorithm': [cluster_algo],
            'Dimensionality Reduction Time (s)': [dr_time],
            'Clustering Time (s)': [cluster_time],
            'Number of Suggested Clusters': [n_clusters],
            'Calinski-Harabasz Index': [ch_index],
            'Davies-Bouldin Index': [db_index],
            'Silhouette Score': [silhouette_avg],
            'Adjusted Rand Index Score': [ari_score]
        })

        # Concatenate the new row with the existing DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)

# Cluster without dimensionality reduction techniques
# Select random images
random_images = select_random_images_by_class(train_images, train_labels)

# Step 7 - Clustering
# Prepare data for clustering
clustering_data = flattened_test_images

for cluster_algo, cluster_func in [
    ('MiniBatch KMeans', apply_minibatch_kmeans),
    ('DBSCAN', apply_dbscan),
    ('Agglomerative', apply_agglomerative)
]:
    # Measure the clustering time
    start_time_cluster = time.time()
    labels = cluster_func(clustering_data)
    end_time_cluster = time.time()
    cluster_time = end_time_cluster - start_time_cluster

    # Calculate the number of clusters (excluding noise for DBSCAN)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Calculate metrics
    ch_index = calinski_harabasz_score(clustering_data, labels)
    db_index = davies_bouldin_score(clustering_data, labels)
    silhouette_avg = silhouette_score(clustering_data, labels)
    ari_score = adjusted_rand_score(test_labels, labels)

    # Create a new DataFrame for the current iteration's results
    new_row = pd.DataFrame({
        'Dimensionality Reduction Technique': "Raw",
        'Clustering Algorithm': [cluster_algo],
        'Dimensionality Reduction Time (s)': 0.0,
        'Clustering Time (s)': [cluster_time],
        'Number of Suggested Clusters': [n_clusters],
        'Calinski-Harabasz Index': [ch_index],
        'Davies-Bouldin Index': [db_index],
        'Silhouette Score': [silhouette_avg],
        'Adjusted Rand Index Score': [ari_score]
    })

    # Concatenate the new row with the existing DataFrame
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Columns for time to be rounded to 3 decimals
time_columns = [
    'Dimensionality Reduction Time (s)',
    'Clustering Time (s)'
]

# Other metric columns to be rounded to 2 decimals
metric_columns = [
    'Calinski-Harabasz Index',
    'Davies-Bouldin Index',
    'Silhouette Score',
    'Adjusted Rand Index Score'
]

# Apply rounding
for col in time_columns:
    results_df[col] = results_df[col].round(3)

for col in metric_columns:
    results_df[col] = results_df[col].round(2)

results_df

# Export the DataFrame to a CSV file
results_df.to_csv('results.csv', index=False)

# Step 10
def show_clustered_images(images, labels, cluster_labels, class_indices, n_images=5):
    """
    Show images of `n_images` for each selected class that are placed into clusters.
    :param images: The array of images.
    :param labels: True labels of the images.
    :param cluster_labels: Labels obtained from clustering.
    :param class_indices: Indices of the classes to be visualized.
    :param n_images: Number of images to display for each class.
    """
    fig, axes = plt.subplots(len(class_indices), n_images, figsize=(n_images * 2, len(class_indices) * 2))

    for i, class_idx in enumerate(class_indices):
        class_images = images[labels == class_idx]
        class_cluster_labels = cluster_labels[labels == class_idx]

        for j in range(n_images):
            # Select a random image from the class that is in the cluster
            cluster = np.random.choice(np.unique(class_cluster_labels))
            image = class_images[np.random.choice(np.where(class_cluster_labels == cluster)[0])]

            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f"Class {class_idx}, Cluster {cluster}")

    plt.tight_layout()
    plt.show()

selected_classes = [0, 4, 8, 9]

# Iterate through each key in the cluster_labels_dict
for key in cluster_labels_dict.keys():
    print(f"Displaying images for {key}")
    show_clustered_images(test_images, test_labels, cluster_labels_dict[key], selected_classes, n_images=4)

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'clustering_data' is your data after dimensionality reduction
neigh = NearestNeighbors(n_neighbors=11)  # Assuming 'min_samples' would be 10 for DBSCAN
neigh.fit(clustering_data)

distances, indices = neigh.kneighbors(clustering_data)

# Sort and plot distances to the k-th nearest neighbors
k_dist = np.sort(distances[:, 10], axis=0)  # 10 is the index of the k-th nearest
plt.figure(figsize=(12, 6))
plt.plot(k_dist)
plt.xlabel('Points sorted by distance to 11th nearest neighbor')
plt.ylabel('11th nearest neighbor distance')
plt.title('k-distance Graph')
plt.show()
