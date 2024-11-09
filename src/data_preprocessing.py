from pandas import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

class DataPreprocessing:
    def __init__(self):
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("DataProcessing") \
            .getOrCreate()

    def load_data(self, file_path, file_format="csv"):
        """
        Load data from a specified file path.
        Args:
            file_path (str): Path to the data file (local or GCS).
            file_format (str): Format of the file, default is CSV.
        Returns:
            DataFrame: Spark DataFrame with loaded data.
        """
        print(f"Loading data from {file_path}...")
        if file_format == "csv":
            df = self.spark.read.csv(file_path, header=False, inferSchema=True)
        elif file_format == "parquet":
            df = self.spark.read.parquet(file_path)
        else:
            raise ValueError("file_format must be either 'csv' or 'parquet'.")

        print(f"Converted {file_path} into Spark DataFrame.")
        df.printSchema()

        # retrieve (non-)numerical columns
        non_list = [item[0] for item in df.dtypes if item[1].startswith('string')]
        num_list = [item[0] for item in df.dtypes if item[1].startswith('string') == False]
        column_list = [non_list, num_list]

        return df, column_list

    def assemble_vector(self, df, column_list):
        """
        Method combines all the final features into a single vector, to make it compatible with KMeans machine learning.
        :param df: Spark DataFrame.
        :param column_list: List of feature names.
        :return: VectorAssembler.
        """
        vec_assembler = VectorAssembler(inputCols=column_list, outputCol="features")
        return vec_assembler.transform(df)

    def transform_features(self, df, column):
        """
        Transform features using encoding and vector assembling.
        Args:
            df (DataFrame): Spark DataFrame to transform.
            categorical_columns (list): List of categorical column names.
            numerical_columns (list): List of numerical column names.
        Returns:
            DataFrame: DataFrame with assembled features ready for scaling.
        """
        print(f"Transform features using encoding and vector assembling...")

        # StringIndexer for categorical columns
        indexers = [StringIndexer(inputCol=column, outputCol=column + "_indexed")]

        # OneHotEncoder for indexed categorical columns
        encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), \
                                  outputCol=column + "_encoded", dropLast=False) \
                    for indexer in indexers]

        # VectorAssembler to combine numerical and encoded categorical features
        assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol=column+"_protocol")

        # Pipeline for transformations
        pipeline =Pipeline(stages = indexers + encoders + [assembler])
        transformed_df = pipeline.fit(df).transform(df)
        return transformed_df.drop(column+'_indexed', column+'_encoded'), column+"_protocol"

    def scale_features(self, df, start_columns):
        """
        Scale features using StandardScaler.
        Args:
            df (DataFrame): DataFrame with assembled features.
            start_columns (list): List of numerical column names that will be scaled.
        Returns:
            DataFrame: DataFrame with scaled features.
        """
        print(f"Scale features using StandardScaler...")
        assembled_col = [col + "_vec" for col in start_columns]
        scaled_col = [col + "_scaled" for col in assembled_col]
        assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in start_columns]
        scalers = [StandardScaler(inputCol=col, outputCol=col + "_scaled") for col in assembled_col]
        pipeline = Pipeline(stages=assemblers + scalers)
        scaledData = pipeline.fit(df).transform(df)

        scaledData = scaledData.drop(*start_columns, *assembled_col)

        return scaledData, scaled_col

