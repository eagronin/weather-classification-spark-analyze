# Analysis

This section describes the analysis of weather patterns in San Diego, CA.  Specifically, we build and evaluate the performance of a decision tree for predicting low humidity days.  Such low humidity days are known to increase the risk of wildfires and, therefore, predicting such days is important for providing a timely warning to the residents and appropriate authorities.

Exploration and cleaning of the data are discussed in the [previous section](https://eagronin.github.io/weather-classification-spark-prepare/).

## Training a Decision Tree Classifier
The following code defines a dataframe with the features used for the decision tree classifier.  It then create the target, a categorical variable to denote if the humidity is not low. If the value is less than 25%, then the categorical value is 0, otherwise the categorical value is 1.  Finally, the code aggregate the features used to make predictions into a single column using `VectorAssembler` and partition the data into training and test data: 

```python
featureColumns = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']

binarizer = Binarizer(threshold = 24.99999, inputCol = "relative_humidity_3pm", outputCol="label")
binarizedDF = binarizer.transform(df)

assembler = VectorAssembler(inputCols = featureColumns, outputCol = "features")
assembled = assembler.transform(binarizedDF)
(trainingData, testData) = assembled.randomSplit([.7,.3], seed = 13234)
```

Next, we careate and train a decision tree:

```python
dt = DecisionTreeClassifier(labelCol = "label", featuresCol = "features", maxDepth = 5, minInstancesPerNode = 20, impurity = "gini")
pipeline = Pipeline(stages = [dt])
model = pipeline.fit(trainingData)
```

Let's make predictions for the test data and compare the target (or label) with its prediction for the first 20 rows of the test dataset:

```python
predictions = model.transform(testData)
predictions.select("prediction", "label").show(20)
```

| prediction | label |
| --- | --- |
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       0.0|  0.0|
|       0.0|  0.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       0.0|  0.0|
|       1.0|  1.0|
|       0.0|  1.0|
|       1.0|  1.0|
|       1.0|  1.0|
|       1.0|  0.0|
|       1.0|  1.0|
|       0.0|  0.0|

The output shows that out of the first 20 target values 18 values are predicted corretly.  

The following code saves the predictions, which are subsequently used for model evaluation:

```python
predictions.select("prediction", "label").coalesce(1).write.save(path = "file:///home/cloudera/Downloads/big-data-4/predictions",
                                                    format = "com.databricks.spark.csv",
                                                    header = 'true')
```

## Evaluation of a Decision Tree Classifier
The following code evaluates the performance of the decision tree:

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = sqlContext.read.load('file:///home/cloudera/Downloads/big-data-4/predictions', 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')
evaluator = MulticlassClassificationEvaluator(
    labelCol = "label",predictionCol = "prediction", metricName = "precision")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))
```

The accuracy of the decision tree perormance using test data is 0.75.

Next, we generate and output the confusion matrix.  

The MulticlassMetrics class can be used to generate a confusion matrix of the classifier above. However, unlike MulticlassClassificationEvaluator, MulticlassMetrics works with RDDs of numbers and not DataFrames, so we need to convert our predictions DataFrame into an RDD.

If we use the RDD attribute of predictions, we see this is an RDD of Rows: `predictions.rdd.take(2)` outputs `[Row(prediction=1.0, label=1.0), Row(prediction=1.0, label=1.0)]`.

Instead, we can map the RDD to tuple to get an RDD of numbers: `predictions.rdd.map(tuple).take(2)` outputs `[(1.0, 1.0), (1.0, 1.0)]`.  The following code then generates the confusion matrix for the decision tree classifier:

```python
from pyspark.mllib.evaluation import MulticlassMetrics

metrics = MulticlassMetrics(predictions.rdd.map(tuple))
metrics.confusionMatrix().toArray().transpose()
```

This results in the following confusion matrix:

```
array([[ 134.,   53.],
       [  29.,  118.]])
```

Previous step: [Data Preparation](https://eagronin.github.io/weather-classification-spark-prepare/)
