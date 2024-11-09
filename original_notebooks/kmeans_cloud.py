import numpy as np
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import time
from pyspark.ml import Pipeline
from tqdm import tqdm

# Initialize Spark Session
spark = SparkSession.builder.appName("KMeansSession").getOrCreate()

data_path = "gs://kmeans_spark_bucket/pro_ass2/kddcup.data"
raw_data = spark.read.csv(data_path, header=False, inferSchema=True)

#we assemble a KMeans-capable dataframe from a "finished" dataframe we already assembled
def assemble_vector(dataframe, columns):
    vec_assembler = VectorAssembler(inputCols=columns, outputCol="features")
    return vec_assembler.transform(dataframe)

#for the first two tasks, we need to drop all non-numeric columns, as kMeans cannot deal with them
def is_numeric_column(column):
    return column[1] != "string"

numeric_columns = []
non_numeric_columns = []

for column in raw_data.dtypes:
    if is_numeric_column(column):
        numeric_columns.append(column[0])
    else:
        non_numeric_columns.append(column[0])

#print(numeric_columns)

#dataset we use in Tasks 1 and 2
#numeric_data = raw_data.drop(*non_numeric_columns)

def scale_dataframe(input_dataframe, start_columns):

    assembled_col = [col+"_vec" for col in start_columns]
    scaled_col = [col+"_scaled" for col in assembled_col]
    assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in start_columns]
    scalers = [StandardScaler(inputCol=col, outputCol=col + "_scaled") for col in assembled_col]
    pipeline = Pipeline(stages=assemblers + scalers)
    scalerModel = pipeline.fit(input_dataframe)
    scaledData = scalerModel.transform(input_dataframe)

    scaledData = scaledData.drop(*start_columns, *assembled_col)

    return scaledData, scaled_col

def one_code_encode(dataframe, column):
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_indexed")]
    encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(), outputCol= column+'_encoded') for indexer in indexers]
    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol=column+"_protocol")

    pipeline = Pipeline(stages=indexers + encoders+[assembler])
    model = pipeline.fit(dataframe)
    transformed = model.transform(dataframe)
    return transformed.drop(column+'_indexed', column+'_encoded'), column+"_protocol"

feature_cols = numeric_columns.copy()
modified_data, col = one_code_encode(raw_data, column='_c1')
feature_cols += [col]
#modified_data, col = one_code_encode(modified_data, column='_c2')
#feature_cols += [col]
#modified_data, col = one_code_encode(modified_data, column='_c3')
#feature_cols += [col]
modified_scaled_data, scaled_col = scale_dataframe(modified_data, feature_cols)
transformed_modified = assemble_vector(modified_scaled_data, scaled_col)

k_from_task_4 = 56
k_to_task_4 = 57
squared_score_task_4 = []
predictions = []

start_time = time.time()
for i in tqdm(range(k_from_task_4, k_to_task_4)):
    kmeans = KMeans(k=i, seed=1)
    model = kmeans.fit(transformed_modified)
    predictions.append(model.transform(transformed_modified))
    score = model.summary.trainingCost
    squared_score_task_4.append(score)

end_time = time.time()
duration = end_time - start_time

def entropy_score(dataframe):

    x = dataframe \
        .groupBy('prediction') \
        .count() \
        .sort('prediction') \
        .toPandas()

    gamma = dataframe \
        .groupBy('prediction', '_c41') \
        .count() \
        .sort('prediction') \
        .toDF('prediction', 'label', 'count').toPandas()

    total_entropy = 0
    for _, rows in x.iterrows():
        cluster_id = rows['prediction']
        amount_objects = rows['count']
        cluster_label_counts = gamma.loc[gamma['prediction'] == cluster_id].values[:, 2].astype(np.float64)
        a = np.divide(cluster_label_counts, amount_objects)
        cluster_sum = np.sum(np.multiply(a, np.log2(a)))
        total_entropy -= cluster_sum * amount_objects / raw_data.count()

    return total_entropy

start_time = time.time()
entropy_list = [entropy_score(i) for i in tqdm(predictions)]
end_time = time.time()
duration_eval = end_time - start_time

print(f"KMeans Execution: {duration} seconds.")
print(f"Evaluation(Entropy) Execution: {duration_eval} seconds.")