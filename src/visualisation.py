import os
import matplotlib.pyplot as plt

class Visualisation:
    def __init__(self, save_plots=False, plot_dir="plots"):
        """
        Initialize the Visualisation class.
        Args:
            save_plots (bool): Whether to save plots instead of displaying them.
            plot_dir (str): Directory to save plots.
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir

        # create plot directory if saving plots
        if self.save_plots:
            os.makedirs(self.plot_dir, exist_ok=True)

    def plot_silhouette_score(self, k_values, silhouette_scores):
        """
        Plot silhouette score against k values to find the optimal number of clusters.
        Args:
            k_values (list): List of k values (number of clusters).
            silhouette_scores (list): List of silhouette scores corresponding to each k value.
        """
        print("Plotting silhouette score vs. number of clusters (k)...")

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs. Number of Clusters")

        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, "silhouette_score.png"))
            print(f"Silhouette Score Plot saved in {self.plot_dir}")
        else:
            plt.show()