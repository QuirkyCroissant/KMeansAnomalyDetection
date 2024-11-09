from tqdm import tqdm

from data_preprocessing import DataPreprocessing
from clustering import Clustering
from visualisation import Visualisation

# Initialize data preprocessing and clustering
data_preprocessor = DataPreprocessing()
df, list = data_preprocessor.load_data("../data/kddcup.data_10_percent")

non_numeric_columns = list[0]
numeric_columns = list[1]

print(f"non numeric columns: {non_numeric_columns}")
print(f"numeric columns: {numeric_columns}")

feature_cols = numeric_columns.copy()
transformed_df, col = data_preprocessor.transform_features(df, "_c1")
feature_cols += [col]
scaled_df, scaled_cols = data_preprocessor.scale_features(transformed_df, feature_cols)
transformed_modified = data_preprocessor.assemble_vector(scaled_df, scaled_cols)

# Clustering and training

k_range = range(45, 85)
k_values = []
silhouette_scores = []

for k in tqdm(k_range):
    print(f"Training and evaluating K-Means model with {k} clusters")
    clustering = Clustering(k=k)
    clustering_df = clustering.train_model(transformed_modified)
    silhouette_score = clustering.evaluate_model(clustering_df)

    k_values.append(k)
    silhouette_scores.append(silhouette_score)

# Visualisations
visualiser = Visualisation(save_plots=True)
visualiser.plot_silhouette_score(k_values, silhouette_scores)

