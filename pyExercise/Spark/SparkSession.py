from pyspark.sql import SparkSession
'''
URL:- https://stackoverflow.com/questions/25481325/how-to-set-up-spark-on-windows
'''
spark = SparkSession \
    .builder \
    .appName("") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()