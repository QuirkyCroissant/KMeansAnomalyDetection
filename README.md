# KMeans Anomaly Detection on KDD Cup Data

This project implements an anomaly detection system using **KMeans clustering** on the **KDD Cup 1999 Network Traffic Dataset**. Originally implemented in Jupyter notebooks, the code has been refactored into a modular Python project with separate classes for data processing, clustering, and visualization. The project is designed to detect network intrusions and anomalies in network traffic data by clustering similar data points together and identifying outliers.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Google Cloud Dataproc Setup](#google-cloud-dataproc-setup)
- [Results](#results)
- [References](#references)

---

## Project Overview

The primary goal of this project is to use unsupervised learning to detect anomalies in network traffic data, which could represent potential intrusions. **KMeans clustering** was chosen due to its ability to group similar connections, enabling the detection of outliers (anomalies) without labeled data. This project uses **Apache Spark** with **PySpark** to handle large datasets efficiently and leverages **Google Cloud Dataproc** for performance scaling.

## Dataset

We use the **KDD Cup 1999 Data Set**:
- **kddcup.data**: Full dataset with approximately 5 million rows and 41 features.
- **kddcup.data_10_percent**: A reduced version (10%) for local development and testing.

The dataset includes both **normal** connections and **network attacks** (e.g., `smurf`, `neptune`, `teardrop`).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/kmeans_anomaly_detection.git
    cd kmeans_anomaly_detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure **Google Cloud SDK** is installed if using Google Cloud Dataproc.

## Project Structure

```plaintext
kmeans_anomaly_detection/
├── data/
│   ├── kddcup.data
│   ├── kddcup.data_10_percent
├── original_notebooks/
│   ├── kmeans_cloud.ipynb
│   ├── kmeans_local.ipynb
│   └── kmeans_real.ipynb
├── src/
│   ├── clustering.py         # KMeans clustering implementation
│   ├── data_processing.py    # Data loading, preprocessing, and scaling
│   ├── main.py               # Main script to run end-to-end clustering
│   ├── plots/                # Directory to save generated plots
│   └── visualization.py      # Visualization of clustering results
├── .gitignore
├── LICENSE
└── README.md
```

## Key Components

- data_processing.py: Handles data loading, normalization, scaling, and one-hot encoding for categorical features.
- clustering.py: Implements KMeans clustering and evaluation with silhouette scores.
- visualization.py: Provides visualizations for clustering results, including an elbow plot for silhouette scores across multiple k values
- main.py: Coordinates data processing, clustering, and visualization, running experiments across a range of k values to identify the optimal cluster count.

## Configuration

In main.py, you can adjust parameters such as:

- k_range: Range of cluster numbers to test for finding the optimal k using the silhouette score.
- save_plots: Set to True to save generated plots in the plots/ directory, or False to display them interactively.

## Google Cloud Dataproc Setup

For large-scale data processing, this project supports running on Google Cloud Dataproc.

1. Upload Data to Google Cloud Storage (GCS): Use gsutil to upload the dataset to a GCS bucket.
```bash
gsutil cp data/kddcup.data gs://your-bucket-name/kddcup.data
```
2. Submit a Dataproc Job: Use the following gcloud command to submit the job to a Dataproc cluster:
```bash
gcloud dataproc jobs submit pyspark src/main.py --cluster=your-cluster-name --region=your-region
```

## Results

Using the elbow method and silhouette scores, we found that an optimal value for k (number of clusters) for the reduced dataset was around k=56. This configuration allowed the model to differentiate between normal and anomalous connections effectively, achieving a notable improvement in accuracy and reducing computation time.
Example Plots

### The plots directory (src/plots/) contains:

- Silhouette Score vs. Number of Clusters (k): Helps identify the optimal k value.
- Cluster Visualizations: Shows the distribution of clustered network connections.

## References

This project builds on concepts discussed in the *Scientific Data Management* course at the University of Vienna, utilizing KMeans++ clustering and Apache Spark. Thanks to the KDD Cup for providing the dataset, and Google Cloud for facilitating large-scale processing with Dataproc.