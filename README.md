# Fashion-MNIST Clustering and Dimensionality Reduction

## Overview
This project is developed by four undergraduate students of the Computer Science Department of the University of Macedonia as part of our Machine Learning class. It focuses on applying various dimensionality reduction techniques and clustering algorithms to the Fashion-MNIST dataset, a collection of 28x28 grayscale images of 10 fashion categories. Our goal is to explore and evaluate the effectiveness of different methods in grouping similar fashion items together, thereby demonstrating the practical applications of machine learning techniques in image classification and clustering.

## Features
- Utilizes TensorFlow to load and preprocess the Fashion-MNIST dataset.
- Applies PCA, Stacked Autoencoder, and LDA for dimensionality reduction.
- Implements clustering with MiniBatch KMeans, DBSCAN, and Agglomerative Clustering algorithms.
- Evaluates clustering performance using metrics like Calinski-Harabasz Index, Davies-Bouldin Index, Silhouette Score, and Adjusted Rand Index Score.
- Visualizes reduced data and clustering results for comprehensive analysis.

## Setup and Running the Project
### Prerequisites
Ensure you have Python 3.x installed along with the following libraries:
- TensorFlow
- NumPy
- Pandas
- Seaborn
- Matplotlib
- scikit-learn
- Keras

### Instructions
1. Clone this repository to your local machine.
2. Install the required dependencies:
   
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn tensorflow keras
   ```
3. Run the script:
   ```bash
   python fashion_mnist_clustering.py
   ```

## Dataset
The Fashion-MNIST dataset is used in this project, consisting of 60,000 training images and 10,000 test images, distributed across 10 categories. Each image is a 28x28 grayscale image, associated with a label from 10 classes.

## Results
The outcomes of the dimensionality reduction and clustering are summarized in a `results.csv` file, which includes the time taken for each process, the number of suggested clusters, and various evaluation metrics. Visualizations of the clustered data provide insights into the grouping effectiveness of each combination of techniques.

## Contributors
- Vasileios Efthymiou
- Nikolaos Papadopoulos
- Georgios Tzelalis
- Stavroula Tsoni

We, the contributors, are undergraduate students of the Computer Science Department of the University of Macedonia, and this project is a part of our coursework in Machine Learning. Our collective effort aims to explore and apply machine learning techniques in understanding and categorizing fashion data, thereby gaining hands-on experience with real-world machine learning and data analysis challenges.


   

