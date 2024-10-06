# kMeans-Clustering
A Python implementation of the K-Means clustering algorithm with silhouette analysis to evaluate cluster quality.

This project implements the K-Means clustering algorithm in Python, providing a method for clustering datasets and evaluating the quality of clustering using silhouette analysis. 
The implementation allows you to run K-Means on your own dataset and plot the silhouette coefficient for various numbers of clusters to help choose the best k-value.

## Features
- Load and preprocess datasets from a CSV file.
- Perform K-Means clustering with customisable number of clusters and iterations.
- Evaluate cluster quality using the silhouette coefficient.
- Visualise the silhouette scores for different cluster sizes.

## Prerequisites
The following Python libraries are required to run the code:
- `numpy`
- `pandas`
- `matplotlib`

## Dataset
The code expects a dataset in CSV format with columns separated by spaces. The first column is ignored (e.g., for non-numeric labels). 
The dataset is **not included** in this repository due to licensing concerns, but you can provide your own dataset.
- The dataset should contain numeric columns representing features.
- The correct file path should be provided when running the script.


