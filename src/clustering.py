from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


class Clustering:
    def __init__(self, k=3, seed=42):
        """
        Initialize the Clustering class with K-Means parameters.
        Args:
            k (int): Number of clusters.
            seed (int): Random seed for reproducibility.
        """
        self.k = k
        self.seed = seed
        self.kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=self.k, seed=self.seed)
        self.model = None

    def train_model(self, df):
        """
        Train the K-Means model on the provided DataFrame.
        Args:
            df (DataFrame): DataFrame with scaled features to cluster.
        Returns:
            DataFrame: DataFrame with cluster predictions.
        """
        print(f"Training K-Means model with {self.k} clusters.")
        self.model = self.kmeans.fit(df)
        clustered_df = self.model.transform(df)
        return clustered_df

    def evaluate_model(self, clustered_df):
        """
        Evaluate the model using Silhouette Score.
        Args:
            clustered_df (DataFrame): DataFrame with clustering predictions.
        Returns:
            float: Silhouette score for the clustering model.
        """
        print("Evaluating the clustering model...")
        evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")
        silhouette_score = evaluator.evaluate(clustered_df)
        print(f"Silhouette score: {silhouette_score}")
        return silhouette_score

    def get_cluster_centres(self):
        """
        Get the cluster centres after training.
        Returns:
            list: List of cluster centres.
        """
        if self.model is None:
            raise ValueError("Model has not been trained! Train the model before getting cluster centers.")

        print("Cluster centres:")
        centres = self.model.clusterCentres()
        for i, centre in enumerate(centres):
            print(f"Centre {i}: {centre}")
        return centres
