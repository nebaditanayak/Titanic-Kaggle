# Databricks notebook source
#Homework4: Kaggle Titanic Challenge
#Submitted by: Nebadita Nayak
#Date: 20th October 2017

#Note: For this project, I have referred Databricks BinaryClassification examples and documentation. Please follow the link below
#https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html

#Extract the data sets into the dataframe

#Load the train data
train = sqlContext.read.format("csv")\
                   .option("header","true")\
                   .option("inferSchema","true")\
                   .load("dbfs:/FileStore/tables/g6rkcc6m1508543693194/train.csv")

#Load the test data
test = sqlContext.read.format("csv")\
                   .option("header","true")\
                   .option("inferSchema","true")\
                   .load("/FileStore/tables/g6rkcc6m1508543693194/test.csv")

train.printSchema()
#test.show()

#Check the count of the records
print(test.count())
print(train.count())


# COMMAND ----------

#Data Cleaning and Data Exploration
train.describe().show()
#display(train)

# COMMAND ----------

#Since from the above step we realized that the age column has null values, we will try to gauge the average age of people belonging to a particular class.

exploreAge = train.where(train["Age"].isNotNull()).groupBy('Pclass').avg('Age')
display(exploreAge)

#Pclass     #AvgAge
#1          #32.92324074074074
#2          #18.177026476578412
#3          #28.09146739130435


# COMMAND ----------

from pyspark.ml.feature import Imputer
#Now we will check for null values in the Age column
NullAgeTrain = train.where(train["Age"].isNull())
#display(NullFaresTrain)

#Check Null fares for Test
NullAgeTest = test.where(test["Age"].isNull())
#display(NullFaresTest)

#From the above queries we will realize that there is one Null value for Test data. That is for passengerID = 1044

#Now we will replace this Null value with the mean value of the column. To do so we will use the Imputer transformer
imputer= Imputer(inputCols=["Age"],outputCols=["modifiedAge"]).fit(test)
test = imputer.transform(test)

#Do it for train too , to maintain the sync of columns in test and train
train = imputer.transform(train)

#Check Null fares for Test after running Imputer
NullAgeTest = test.where(test["Age"].isNull())
NullAgeTest.show()

# COMMAND ----------

#I am including this step in my second iteration. I discovered that my Vector Assembler was erroring out in the pipeline because there were null categorical values in the
#Error received: Caused by: org.apache.spark.SparkException: Values to assemble cannot be null.
#Embarked column for for Passenger ID 62 and 830.

exploreEmbarkNull = train.where(train["Embarked"].isNull())
exploreEmbarkNull.show()


# COMMAND ----------

# After discovering the null values, I want to replace them with the highest frequency of Embarked class occuring in the train data.
#From the below query, I realized tha
#Embarked   #Count
#Q          77
#C          168
#S          644

#So, will replace the two null embarked values [Passenger ID: 62,830] with Embarked Values "S" as frequency of null values

exploreEmbark = train.where(train["Embarked"].isNotNull()).groupBy('Embarked').count()
display(exploreEmbark)

# COMMAND ----------

##Replae the Embarked valuees to 'S' for passenger 62 and 830
train = train.na.fill({'Embarked':'S'})
test = test.na.fill({'Embarked':'S'})

#Check the values
embarkedNullRemoved = train.where(train["Embarked"].isNull())
embarkedNullRemoved.show()

# COMMAND ----------

from pyspark.ml.feature import Imputer
#Now we will check for null values in the Fare column
NullFaresTrain = train.where(train["Fare"].isNull())
#display(NullFaresTrain)

#Check Null fares for Test
NullFaresTest = test.where(test["Fare"].isNull())
#display(NullFaresTest)

#From the above queries we will realize that there is one Null value for Test data. That is for passengerID = 1044

#Now we will replace this Null value with the mean value of the column. To do so we will use the Imputer transformer
imputer= Imputer(inputCols=["Fare"],outputCols=["modifiedFare"]).fit(test)
test = imputer.transform(test)

#Do it for train too , to maintain the sync of columns in test and train
train = imputer.transform(train)

#Check Null fares for Test after running Imputer
NullRemovedFaresTest = test.where(test["Fare"].isNull())
NullRemovedFaresTest.show()

# COMMAND ----------

# Rename survived column to "label"
train = train.withColumnRenamed("Survived", "label")
test = test.withColumnRenamed("Survived", "label")

train.show()
test.show()

# COMMAND ----------

from pyspark.ml.feature import *
from pyspark.sql.functions import *

#List of Categorical Variables
#1) Survived: Label 1 or 0
#2) Pclass: Categorical (1,2,3)
#3) Sex: Categorical (Male,Female)
#4) Age: Continuous
#5) SibSp: Categorical(0,1,2,...8) ---Discrete
#6) Parch: Categorical(0,1,2,3...9)----Discrete
#7) Fare: Continuous
#8) Embarked: Categorical(C,Q,S)


#Indexed all the labels, to add the metadata
genderIndex = StringIndexer(inputCol = "Sex", outputCol="indexedSex" ).setHandleInvalid("skip")
embarkIndex = StringIndexer(inputCol="Embarked", outputCol="indexedEmbarked").setHandleInvalid("skip")

#After multiple iterations I realized that we cannot use OHE on features with different categories in the test and the train data set. Because the Vector assembler will error out in the Pipeline.tranform() step due to mismatch in the vector size. Hence we are only using OHE on the gender and Embarked category as their number of categories is constant in the test and in the train data set.

#Creating One hot encoding on indexed columns to produce sparse vectors
#surviveEncode = OneHotEncoder(inputCol="Survived",outputCol="survivedVector")
genderEncode = OneHotEncoder(inputCol=genderIndex.getOutputCol(),outputCol="SexVector")
embarkEncode = OneHotEncoder(inputCol=embarkIndex.getOutputCol(),outputCol="EmbarkedVector")
pClassEncode = OneHotEncoder(inputCol="Pclass", outputCol="PclassVector")
SibSpEncode = OneHotEncoder(inputCol="SibSp",outputCol="SibSpVector")
ParchEncode = OneHotEncoder(inputCol="Parch",outputCol="ParchVector")



# COMMAND ----------

# Transform all features into a vector using VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

numericalCols = ["modifiedAge","modifiedFare"]
#categoricalCols = ["Pclass","Sex","SibSp","Parch","Embarked"]
categoricalCols = ["Sex","Embarked"]
otherCategoricalCols = ["Pclass","SibSp","Parch"]
#assemblerInputs = map(lambda c: c + "Vector", categoricalCols) + numericalCols
assemblerInputs = map(lambda c: c + "Vector", categoricalCols) + numericalCols +otherCategoricalCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")


# COMMAND ----------

#Check the status of the nulls in the data before fitting a model. If everything looks ok, then move ahead.
print(test.select([count(when( col(c).isNull(), c)).alias(c) for c in test.columns]).show())

# COMMAND ----------

#First we will fit a Logistic Regression model for this Binary Classification problem. That is to predict whether a person survived or not (0,1).
from pyspark.ml.classification import LogisticRegression

(trainingData, testData) = train.randomSplit([0.7, 0.3], seed = 100)

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)


# COMMAND ----------

#Execute the model
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[genderIndex,embarkIndex,genderEncode,embarkEncode,pClassEncode,SibSpEncode,ParchEncode,assembler,lr])
#pipeline = Pipeline(stages=[genderIndex,embarkIndex,genderEncode,embarkEncode,pClassEncode,SibSpEncode,ParchEncode])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)

#Check the schema
predictions.printSchema()

# COMMAND ----------

#Now we will evalute our model
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)

# COMMAND ----------

#Now we will try tuning the model with the ParamGridBuilder and the CrossValidator.
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

from pyspark.ml import Pipeline
# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

pipelineforEval = Pipeline(stages=[genderIndex,embarkIndex,genderEncode,embarkEncode,pClassEncode,SibSpEncode,ParchEncode,assembler]).fit(train)
cvData = pipelineforEval.transform(train)
# Run cross validations
cvModel = cv.fit(cvData.select("label","features"))


# COMMAND ----------

# Use test set here so we can measure the accuracy of our model on new data

cvData = pipelineforEval.transform(testData)
predictions = cvModel.transform(cvData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

#Download the CSV for test Data
cvData = pipelineforEval.transform(test)
predictions = cvModel.transform(cvData)
#predictions.printSchema()
display(predictions.select("PassengerId","prediction"))

# COMMAND ----------

#Now we will train our model using a random Forest Classifier
#Train a random classifier
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rf = RandomForestClassifier(labelCol="label",featuresCol="features")
rfPipeline = Pipeline(stages=[genderIndex,embarkIndex,genderEncode,embarkEncode,pClassEncode,SibSpEncode,ParchEncode,assembler,rf])

rfModel = rfPipeline.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = rfModel.transform(testData)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
rfCV = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

cvData = pipelineforEval.transform(trainingData)
# Run cross validations
cvModelRF = rfCV.fit(cvData.select("label","features"))

# COMMAND ----------

# Use test set here so we can measure the accuracy of our model on new data
cvDataTest = pipelineforEval.transform(testData)

rfPredictions = cvModelRF.transform(cvDataTest.select("label","features"))
evaluator.evaluate(predictions)

# COMMAND ----------

#Select the best Model
bestModel = cvModelRF.bestModel

# COMMAND ----------

#Make predictions for the entire test data set
cvDataTest = pipelineforEval.transform(test)
finalPredictions = bestModel.transform(cvDataTest)

# COMMAND ----------

display(finalPredictions.select("passengerID","prediction"))

# COMMAND ----------


