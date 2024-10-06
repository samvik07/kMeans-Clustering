# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(fname):
    """
    This function loads the dataset from a given file.
    Parameters:
    fname (str): The name of the given file.

    Returns:
    numpy.ndarray: A NumPy array containing the loaded dataset.
    """

    try:
        # Load the dataset
        df = pd.read_csv(fname, delimiter=' ')

        # Check if the dataframe is empty
        if df.empty:
            raise ValueError("The dataset is empty.")

        # Check if there's only one data point in the dataset
        if len(df) == 1:
            raise ValueError("There is only one data point in the dataset.")

        # Drop the first column containing non-numeric values
        data = df.drop(columns=[df.columns[0]]).dropna()

        # Check if the dataset is empty after dropping columns
        if data.empty:
            raise ValueError("The dataset contains no valid data after preprocessing.")

        data_array = data.values.astype(float)
        return data_array

    # Exception handling
    except FileNotFoundError:
        raise FileNotFoundError("File not found. Please provide a valid file name.")

    except pd.errors.ParserError:
        raise ValueError("File is corrupted or not in the expected format.")

    except Exception as e:
        raise ValueError(f"An unexpected error occured: {e}")


def computeDistance(a, b):
    """
    This function computes the Euclidean distance between two points.
    Parameters:
    a (numpy.ndarray): The first point represented as a NumPy array.
    b (numpy.ndarray): The second point represented as a NumPy array.

    Returns:
    float: The Euclidean distance between points 'a' and 'b'.
    """

    # Return the Euclidean distance between a and b
    return np.linalg.norm(a-b)


def initialSelection(x, k):
    """
    This function initialises random centroids for K-means clustering.
    Parameters:
    - x (numpy.ndarray): Input data points.
    - k (int): Number of clusters.

    Returns:
    numpy.ndarray: Initial centroids for the clusters.
    """

    # Random seed value to reproduce results
    np.random.seed(45)
    centroids_idx = np.random.choice(len(x), k, replace=False)
    centroids = x[centroids_idx]
    return centroids


def computeClusterRepresentatives(grouped_clusters):
    """
    This function computes the representative points for each cluster.
    Parameters:
    - grouped_clusters (pandas.core.groupby.DataFrameGroupBy): Grouped clusters.

    Returns:
    numpy.ndarray: Representative points for each cluster.
    """

    return grouped_clusters.mean().values


def assignClusterIds(x, centroids):
    """
    This function assigns cluster IDs to data points based on their nearest centroids.
    Parameters:
    - x (numpy.ndarray): Input data points.
    - centroids (numpy.ndarray): Centroids of the clusters.

    Returns:
    list: List of cluster IDs assigned to each data point.
    """

    cluster_ids = []
    for point in x:
        # calculate distances from current point to all centroids
        distances = [computeDistance(point, centroid) for centroid in centroids]
        # find the one with minimum distance and assign it
        cluster_id = np.argmin(distances)
        cluster_ids.append(cluster_id)
    return cluster_ids


def kMeans(x, k, maxIter):
    """
    This function performs the K-means clustering on the input data.
    Parameters:
    - x (numpy.ndarray): Input data points.
    - k (int): Number of clusters.
    - maxIter (int): Maximum number of iterations.

    Returns:
    list: List of cluster IDs assigned to each data point.
    """

    # Initialize centroids
    centroids = initialSelection(x, k)

    for _ in range(maxIter):

        # Assign each point to the nearest centroid
        cluster_ids = assignClusterIds(x, centroids)

        # Create a DataFrame to hold the data points and their assigned clusters
        clusters = pd.DataFrame(x)
        clusters["cluster_id"] = cluster_ids

        # Compute new centroids
        centroids_new = computeClusterRepresentatives(clusters.groupby("cluster_id"))

        # Check convergence
        if np.array_equal(centroids, centroids_new):
            break

        # Update centroids
        centroids = centroids_new

    # Return final clusters
    clusters = clusters.to_numpy()
    return cluster_ids


def distanceMatrix(dataset, dist=computeDistance):
    """
    This function computes the distance matrix for a given dataset.
    Parameters:
    - dataset (list): List of data points.
    - dist (function): Function to compute distance between data points. Default is Euclidean distance.

    Returns:
    numpy.ndarray: Distance matrix.
    """

    # Compute the number of objects in the dataset
    N = len(dataset)

    # Distance matrix
    distMatrix = np.zeros((N, N))
    # Compute pairwise distances between the objects
    for i in range(N):
        for j in range(N):
            # Distance is symmetric, so compute the distances between i and j only once
            if i < j:
                distMatrix[i][j] = dist(dataset[i], dataset[j])
                distMatrix[j][i] = distMatrix[i][j]
    return distMatrix


def computeSilhouttee(dataset, clusters, distMatrix):
    """
    This function computes the silhouette coefficient for a given clustering.
    Parameters:
    - dataset (list): List of data points.
    - clusters (list): List of cluster IDs assigned to each data point.
    - distMatrix (numpy.ndarray): Distance matrix.

    Returns:
    float: Silhouette coefficient.
    """

    N = len(dataset)
    silhouette = np.zeros(N)
    a = np.zeros(N)
    b = np.full(N, np.inf)

    for i in range(N):
        # Get the cluster ID for the current data point
        cluster_id = clusters[i]

        # Compute a(i) within same cluster
        a[i] = np.mean(distMatrix[i, clusters == cluster_id])

        # Compute b(i) for nearest different cluster
        for j in set(clusters):
            if j != cluster_id:
                temp_b = np.mean(distMatrix[i, clusters == j])
                b[i] = min(b[i], temp_b)

        # Compute silhouette coefficients
        if a[i] == 0 or b[i] == np.inf:
            silhouette[i] = 0  # Silhouette coefficient is 0 for single data points
        else:
            silhouette[i] = (b[i] - a[i]) / max(a[i], b[i])
    silhouetteCoeff = np.mean(silhouette)
    return silhouetteCoeff


def plotSilhouette(dataset, distMatrix, k_max):
    """
    The function plots the silhouette coefficients for different numbers of clusters.
    Parameters:
    - dataset (list): List of data points.
    - distMatrix (numpy.ndarray): Distance matrix.
    - k_max (int): Maximum number of clusters to consider.

    Returns: None
    """

    clusteringSilhouette = {}

    for k in range(1, k_max+1):
        # Perform K-Means clustering
        clusters = kMeans(dataset, k, maxIter=10)
        # Compute silhouette coefficients
        clusteringSilhouette[k] = computeSilhouttee(dataset, clusters, distMatrix)

    # Plot the silhouette coefficients with respect to the number of clusters k
    plt.bar(range(1, k_max+1), list(clusteringSilhouette.values()), align='center', label='K-Means')
    plt.xticks(range(1, k_max+1))
    plt.title("Silhouette Coefficients for different Numbers of Clusters", fontsize=16)
    plt.xlabel('Number of clusters (k)', fontsize=14)
    plt.ylabel('Silhouette coefficient', fontsize=14)

    values = list(clusteringSilhouette.values())
    # Annotate each bar with its value
    for i, value in enumerate(values):
        plt.text(i+1, value, '{:.4f}'.format(value), ha='center', va='bottom')
    plt.legend()

    # Save the plot in the current directory
    plt.savefig('silhouette_plot_kMeans.png')


def main():
    """
    This is the main function which calls the various other functions like
    load_dataset, distanceMatrix and plotSilhouette functions.
    Parameters:None
    Returns:None
    """

    # Make sure to replace 'dataset' with your own dataset file path.
    dataset = load_dataset("dataset")
    distMatrix = distanceMatrix(dataset, dist=computeDistance)
    plotSilhouette(dataset, distMatrix, k_max=9)


if __name__ == "__main__":
    main()
