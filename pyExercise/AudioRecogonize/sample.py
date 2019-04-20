from pyspark import SparkContext

logFile = "D:\Workspace\pyExercise\daily_show_guests.csv"

sc = SparkContext("spark://localhost:4040", "first app")
# raw_data = sc.textFile(file_name)
# raw_data.take(5)

logData = sc.textFile(logFile).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
print("Lines with a: %i, lines with b: %i" % (numAs, numBs))