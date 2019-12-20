
**Guojing Wu** | UNI: gw2383 | *2019-12-03*

# P9120 Final Project

This is for modeling and predicting


```python
from pyspark.sql import SparkSession    
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import IDF
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import json, random
from pyspark.sql.types import IntegerType

random.seed(100)

spark = SparkSession\
    .builder\
    .appName("WhatsCook")\
    .getOrCreate()

# transform the output label to cuisine name
list_cuisine = [("greek", 0),
               ("southern_us", 1),
               ("filipino", 2),
               ("indian", 3),
               ("jamaican", 4),
               ("spanish", 5),
               ("italian", 6),
               ("mexican", 7),
               ("chinese", 8),
               ("british", 9),
               ("thai", 10),
               ("vietnamese", 11),
               ("cajun_creole", 12),
               ("brazilian", 13),
               ("french", 14),
               ("japanese", 15),
               ("irish", 16),
               ("korean", 17),
               ("moroccan", 18), 
               ("russian", 19)]
table_cuisine = spark.createDataFrame(list_cuisine, ["cuisine", "label"])

traindata = spark.read.format("json").load("gs://big_data_hw/P9120/training.json")
testdata = spark.read.format("json").load("gs://big_data_hw/P9120/testing.json")

traindata.show(5)
```

    +-----------+-----+--------------------+-----+
    |    cuisine|   id|         ingredients|label|
    +-----------+-----+--------------------+-----+
    |      greek|10259|romaine-lettuce, ...|    0|
    |southern_us|25693|plain-flour, grou...|    1|
    |   filipino|20130|eggs, pepper, sal...|    2|
    |     indian|22213|water, vegetable-...|    3|
    |     indian|13162|black-pepper, sha...|    3|
    +-----------+-----+--------------------+-----+
    only showing top 5 rows
    


## Tokenizer then Word2Vec or TF-IDF

* Word2vec: Word2vec is a method of computing vector representations of words introduced by a team of researchers at Google

* TF-IDF: 


```python
regexTokenizer = RegexTokenizer(inputCol="ingredients", outputCol="words", pattern="[^A-Za-z]+", toLowercase=True)

# word2vec
word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="words", outputCol="features")
pipeline_w2v = Pipeline(stages=[regexTokenizer, word2Vec])
pipeline_w2v_model = pipeline_w2v.fit(traindata)
train_w2v = pipeline_w2v_model.transform(traindata)
test_w2v = pipeline_w2v_model.transform(testdata)

# TF-IDF
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=20)
idf = IDF(inputCol="raw_features", outputCol="features")
pipeline_tfidf = Pipeline(stages=[regexTokenizer, hashingTF, idf])
pipeline_tfidf_model_train = pipeline_tfidf.fit(traindata)
train_tfidf_ready = pipeline_tfidf_model_train.transform(traindata)
test_tfidf = pipeline_tfidf_model_train.transform(testdata)
```


```python
train_w2v.show()
```

    +-----------+-----+--------------------+-----+--------------------+--------------------+
    |    cuisine|   id|         ingredients|label|               words|            features|
    +-----------+-----+--------------------+-----+--------------------+--------------------+
    |      greek|10259|romaine-lettuce, ...|    0|[romaine, lettuce...|[0.01136476354440...|
    |southern_us|25693|plain-flour, grou...|    1|[plain, flour, gr...|[0.02387200678257...|
    |   filipino|20130|eggs, pepper, sal...|    2|[eggs, pepper, sa...|[-0.0544260346330...|
    |     indian|22213|water, vegetable-...|    3|[water, vegetable...|[-0.0371666610240...|
    |     indian|13162|black-pepper, sha...|    3|[black, pepper, s...|[-0.0583169031606...|
    |   jamaican| 6602|plain-flour, suga...|    4|[plain, flour, su...|[0.00514449183829...|
    |    spanish|42779|olive-oil, salt, ...|    5|[olive, oil, salt...|[-0.0185615147978...|
    |    italian| 3735|sugar, pistachio-...|    6|[sugar, pistachio...|[0.01409102841797...|
    |    mexican|16903|olive-oil, purple...|    7|[olive, oil, purp...|[0.05111392151564...|
    |    italian|12734|chopped-tomatoes,...|    6|[chopped, tomatoe...|[-0.0230685837034...|
    |    italian| 5875|pimentos, sweet-p...|    6|[pimentos, sweet,...|[-0.0102291418535...|
    |    chinese|45887|low-sodium-soy-sa...|    8|[low, sodium, soy...|[-0.0309847123610...|
    |    italian| 2698|Italian-parsley-l...|    6|[italian, parsley...|[-0.1018889651254...|
    |    mexican|41995|ground-cinnamon, ...|    7|[ground, cinnamon...|[-0.0102933945846...|
    |    italian|31908|fresh-parmesan-ch...|    6|[fresh, parmesan,...|[0.03115076652417...|
    |     indian|24717|tumeric, vegetabl...|    3|[tumeric, vegetab...|[-0.1260891907149...|
    |    british|34466|greek-yogurt, lem...|    9|[greek, yogurt, l...|[-0.1128140973991...|
    |    italian| 1420|italian-seasoning...|    6|[italian, seasoni...|[-0.0338676186899...|
    |       thai| 2941|sugar, hot-chili,...|   10|[sugar, hot, chil...|[-0.2320626805303...|
    | vietnamese| 8152|soy-sauce, vegeta...|   11|[soy, sauce, vege...|[-0.0176983608731...|
    +-----------+-----+--------------------+-----+--------------------+--------------------+
    only showing top 20 rows
    



```python
train_w2v.take(1)
```




    [Row(cuisine='greek', id=10259, ingredients='romaine-lettuce, black-olives, grape-tomatoes, garlic, pepper, purple-onion, seasoning, garbanzo-beans, feta-cheese-crumbles', label=0, words=['romaine', 'lettuce', 'black', 'olives', 'grape', 'tomatoes', 'garlic', 'pepper', 'purple', 'onion', 'seasoning', 'garbanzo', 'beans', 'feta', 'cheese', 'crumbles'], features=DenseVector([0.0114, -0.1702, -0.2332, 0.0414, -0.0933, -0.0583, -0.1135, 0.095, 0.1387, 0.0968, -0.2169, -0.0481, 0.071, -0.1068, -0.1895, 0.2072, 0.0295, 0.0662, 0.0819, -0.1132, 0.082, -0.1849, -0.0187, 0.0831, 0.0498, -0.0231, 0.0379, -0.0934, 0.0628, -0.1878, -0.03, -0.1295, 0.09, 0.1197, 0.0098, 0.0747, 0.1344, 0.1693, 0.0377, 0.0612, 0.1548, -0.0068, -0.0378, 0.0417, 0.0775, -0.0962, 0.0149, -0.2954, -0.0855, 0.0166, 0.0958, -0.0355, 0.1022, -0.1772, 0.1561, -0.0782, 0.1287, -0.1739, 0.0358, 0.0888, -0.0002, -0.032, -0.139, -0.1683, -0.1399, 0.0683, 0.0726, 0.0851, 0.0374, 0.0664, 0.0062, -0.1135, 0.0624, 0.2568, 0.0, -0.0275, 0.0274, 0.0497, 0.1231, 0.0324, -0.0426, -0.092, -0.0946, -0.0056, -0.0352, -0.0141, 0.1934, -0.0824, -0.0338, 0.1033, -0.0723, -0.1599, -0.0083, 0.0047, -0.1244, -0.0754, 0.0638, -0.0807, 0.1038, 0.0716]))]




```python
train_tfidf_ready.show()
```

    +-----------+-----+--------------------+-----+--------------------+--------------------+--------------------+
    |    cuisine|   id|         ingredients|label|               words|        raw_features|            features|
    +-----------+-----+--------------------+-----+--------------------+--------------------+--------------------+
    |      greek|10259|romaine-lettuce, ...|    0|[romaine, lettuce...|(20,[3,4,5,6,7,9,...|(20,[3,4,5,6,7,9,...|
    |southern_us|25693|plain-flour, grou...|    1|[plain, flour, gr...|(20,[3,4,5,7,8,9,...|(20,[3,4,5,7,8,9,...|
    |   filipino|20130|eggs, pepper, sal...|    2|[eggs, pepper, sa...|(20,[3,4,5,7,8,9,...|(20,[3,4,5,7,8,9,...|
    |     indian|22213|water, vegetable-...|    3|[water, vegetable...|(20,[3,4,9,10,16]...|(20,[3,4,9,10,16]...|
    |     indian|13162|black-pepper, sha...|    3|[black, pepper, s...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|
    |   jamaican| 6602|plain-flour, suga...|    4|[plain, flour, su...|(20,[0,3,5,6,7,9,...|(20,[0,3,5,6,7,9,...|
    |    spanish|42779|olive-oil, salt, ...|    5|[olive, oil, salt...|(20,[0,1,2,3,4,5,...|(20,[0,1,2,3,4,5,...|
    |    italian| 3735|sugar, pistachio-...|    6|[sugar, pistachio...|(20,[0,3,4,7,9,10...|(20,[0,3,4,7,9,10...|
    |    mexican|16903|olive-oil, purple...|    7|[olive, oil, purp...|(20,[0,1,3,4,5,6,...|(20,[0,1,3,4,5,6,...|
    |    italian|12734|chopped-tomatoes,...|    6|[chopped, tomatoe...|(20,[2,3,4,5,8,9,...|(20,[2,3,4,5,8,9,...|
    |    italian| 5875|pimentos, sweet-p...|    6|[pimentos, sweet,...|(20,[1,3,4,6,7,8,...|(20,[1,3,4,6,7,8,...|
    |    chinese|45887|low-sodium-soy-sa...|    8|[low, sodium, soy...|(20,[0,2,3,4,5,7,...|(20,[0,2,3,4,5,7,...|
    |    italian| 2698|Italian-parsley-l...|    6|[italian, parsley...|(20,[3,4,7,8,9,10...|(20,[3,4,7,8,9,10...|
    |    mexican|41995|ground-cinnamon, ...|    7|[ground, cinnamon...|(20,[0,2,3,4,5,6,...|(20,[0,2,3,4,5,6,...|
    |    italian|31908|fresh-parmesan-ch...|    6|[fresh, parmesan,...|(20,[0,2,3,5,6,7,...|(20,[0,2,3,5,6,7,...|
    |     indian|24717|tumeric, vegetabl...|    3|[tumeric, vegetab...|(20,[0,2,4,5,7,10...|(20,[0,2,4,5,7,10...|
    |    british|34466|greek-yogurt, lem...|    9|[greek, yogurt, l...|(20,[0,4,8,9,15,1...|(20,[0,4,8,9,15,1...|
    |    italian| 1420|italian-seasoning...|    6|[italian, seasoni...|(20,[1,3,4,9,15,1...|(20,[1,3,4,9,15,1...|
    |       thai| 2941|sugar, hot-chili,...|   10|[sugar, hot, chil...|(20,[0,4,9,11,15,...|(20,[0,4,9,11,15,...|
    | vietnamese| 8152|soy-sauce, vegeta...|   11|[soy, sauce, vege...|(20,[0,1,3,4,5,7,...|(20,[0,1,3,4,5,7,...|
    +-----------+-----+--------------------+-----+--------------------+--------------------+--------------------+
    only showing top 20 rows
    



```python
train_tfidf_ready.select("features").take(1)
```




    [Row(features=SparseVector(20, {3: 0.2834, 4: 0.2814, 5: 0.9641, 6: 2.0477, 7: 0.3063, 9: 0.403, 13: 0.3856, 15: 1.0119, 17: 0.4117}))]



PCA

## One vs. Rest SVM：

Reduction of Multiclass Classification to Binary Classification. Performs reduction using one against all strategy. For a multiclass classification with k classes, train k models (one per class). Each example is scored against all k models and the model with highest score is picked to label the example.


```python
svc = LinearSVC(maxIter = 1000, tol=0.001, aggregationDepth=3, regParam=0.01)
ovr_svc = OneVsRest(classifier=svc)
ovr_svc_model = ovr_svc.fit(train_w2v)
prediction_w2v_ovr = ovr_svc_model.transform(test_w2v)

prediction_w2v_ovr.show()
```

    +-----+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|            features|prediction|
    +-----+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|[-0.0231250560842...|       1.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|[0.05239258537767...|      14.0|
    |41580|sausage-links, fe...|[sausage, links, ...|[-0.0424788616597...|       6.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|[-0.0076890995592...|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|[0.00895711896009...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|[-0.0527554646444...|       1.0|
    |19666|grape-juice, oran...|[grape, juice, or...|[-0.0991512559354...|       7.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|[-0.0522039992614...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|[0.07571529619218...|       7.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|[0.02974281460046...|       1.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|[-0.0378114992850...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|[-0.0036597959096...|       6.0|
    |42969|curry-powder, gro...|[curry, powder, g...|[0.01816459600296...|       3.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|[-0.0403843747141...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|[-0.0262535809539...|       1.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|[-0.0219391180202...|      14.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|[0.02341907822327...|       1.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|[-0.0503853579311...|       1.0|
    |19001|chopped-green-chi...|[chopped, green, ...|[0.07742244796827...|       7.0|
    |16526|cold-water, chick...|[cold, water, chi...|[0.00318516012938...|       1.0|
    +-----+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
prediction_w2v_ovr = prediction_w2v_ovr \
    .withColumn("prediction", prediction_w2v_ovr["prediction"].cast(IntegerType()))
prediction_w2v_ovr_output = prediction_w2v_ovr \
    .join(table_cuisine, prediction_w2v_ovr.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

prediction_w2v_ovr_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_w2v_ovr.csv")
```


```python
svc = LinearSVC(maxIter = 1000, tol=0.001, aggregationDepth=3, regParam=0.01)
ovr_svc = OneVsRest(classifier=svc)
ovr_svc_model = ovr_svc.fit(train_tfidf_ready)
prediction_tfidf_ovr = ovr_svc_model.transform(test_tfidf)

prediction_tfidf_ovr.show()
```

    +-----+--------------------+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|        raw_features|            features|prediction|
    +-----+--------------------+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|(20,[0,3,7,8,9,10...|(20,[0,3,7,8,9,10...|       1.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|(20,[0,2,3,6,9,12...|(20,[0,2,3,6,9,12...|       7.0|
    |41580|sausage-links, fe...|[sausage, links, ...|(20,[1,4,11,14,17...|(20,[1,4,11,14,17...|      14.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|(20,[1,3,4,5,7,8,...|(20,[1,3,4,5,7,8,...|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|(20,[1,2,3,4,5,6,...|(20,[1,2,3,4,5,6,...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|(20,[0,2,3,6,7,8,...|(20,[0,2,3,6,7,8,...|       7.0|
    |19666|grape-juice, oran...|[grape, juice, or...|(20,[6,7,13,17],[...|(20,[6,7,13,17],[...|       6.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|(20,[0,2,3,4,5,7,...|(20,[0,2,3,4,5,7,...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|       7.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|       7.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|(20,[1,3,4,5,6,7,...|(20,[1,3,4,5,6,7,...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|(20,[0,5,6,7,8,9,...|(20,[0,5,6,7,8,9,...|       6.0|
    |42969|curry-powder, gro...|[curry, powder, g...|(20,[0,2,3,5,8,9,...|(20,[0,2,3,5,8,9,...|       3.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|(20,[1,3,4,6,9,10...|(20,[1,3,4,6,9,10...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|(20,[0,3,7,9,12,1...|(20,[0,3,7,9,12,1...|       6.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|(20,[0,3,4,6,8,9,...|(20,[0,3,4,6,8,9,...|       6.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|(20,[1,3,5,6,7,11...|(20,[1,3,5,6,7,11...|       6.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|(20,[3,4,7,11,14,...|(20,[3,4,7,11,14,...|       6.0|
    |19001|chopped-green-chi...|[chopped, green, ...|(20,[3,5,6,7,8,9,...|(20,[3,5,6,7,8,9,...|       7.0|
    |16526|cold-water, chick...|[cold, water, chi...|(20,[0,3,4,7,8,9,...|(20,[0,3,4,7,8,9,...|      18.0|
    +-----+--------------------+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
prediction_tfidf_ovr = prediction_tfidf_ovr \
    .withColumn("prediction", prediction_tfidf_ovr["prediction"].cast(IntegerType()))
prediction_tfidf_ovr_output = prediction_tfidf_ovr \
    .join(table_cuisine, prediction_tfidf_ovr.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

prediction_tfidf_ovr_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_tfidf_ovr")
```

## LogisticRegression Model


```python
lr = LogisticRegression(maxIter=1000, regParam=0.001)
lrModel_w2v = lr.fit(train_w2v)
prediction_w2v_lr = lrModel_w2v.transform(test_w2v)
```


```python
# show detail of the model
lrModel_w2v.coefficientMatrix
```




    DenseMatrix(20, 100, [0.9042, -4.0555, -1.5419, 3.491, -2.6237, -3.4607, -4.7838, -1.4499, ..., -0.7345, -0.7172, 0.859, 2.2187, 1.6635, -0.2341, -3.1002, -0.7367], 1)




```python
prediction_w2v_lr.show()
```

    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|[-0.0231250560842...|[0.32156966999207...|[0.02066937219587...|       1.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|[0.05239258537767...|[-4.5929916356842...|[9.01950099252958...|      14.0|
    |41580|sausage-links, fe...|[sausage, links, ...|[-0.0424788616597...|[4.42680845063670...|[0.07113243664761...|       6.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|[-0.0076890995592...|[-1.6206459497995...|[5.1640631958258E...|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|[0.00895711896009...|[4.18116142557791...|[0.00510893986042...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|[-0.0527554646444...|[-1.5555952310810...|[6.07665208075470...|       1.0|
    |19666|grape-juice, oran...|[grape, juice, or...|[-0.0991512559354...|[3.68755160772735...|[0.10695718633486...|       5.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|[-0.0522039992614...|[-2.8391563518050...|[1.69845345468433...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|[0.07571529619218...|[1.08485145298836...|[0.00165945988749...|       7.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|[0.02974281460046...|[-1.8668381925496...|[0.00154833264108...|       9.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|[-0.0378114992850...|[2.3974593855648,...|[0.00461438523966...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|[-0.0036597959096...|[5.81536999533831...|[0.59317068730371...|       0.0|
    |42969|curry-powder, gro...|[curry, powder, g...|[0.01816459600296...|[1.75402815937704...|[0.00138663106152...|       3.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|[-0.0403843747141...|[3.20476728879103...|[1.35767101183982...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|[-0.0262535809539...|[-5.7212913835503...|[1.80631798839386...|       1.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|[-0.0219391180202...|[2.38911956473570...|[0.15663083831733...|      14.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|[0.02341907822327...|[-3.8490624305017...|[5.90117853017129...|       1.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|[-0.0503853579311...|[-3.1940223040321...|[1.17150544891076...|       1.0|
    |19001|chopped-green-chi...|[chopped, green, ...|[0.07742244796827...|[2.30059826635625...|[0.00621471926798...|       7.0|
    |16526|cold-water, chick...|[cold, water, chi...|[0.00318516012938...|[-1.4293295021057...|[7.83417318719048...|       1.0|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
prediction_w2v_lr.select("probability").take(1)
```




    [Row(probability=DenseVector([0.0207, 0.3147, 0.0073, 0.013, 0.0059, 0.0071, 0.0392, 0.0562, 0.0075, 0.2511, 0.0006, 0.0003, 0.0137, 0.0172, 0.0162, 0.0273, 0.078, 0.0108, 0.0015, 0.1117]))]




```python
prediction_w2v_lr = prediction_w2v_lr \
    .withColumn("prediction", prediction_w2v_lr["prediction"].cast(IntegerType()))
prediction_w2v_lr_output = prediction_w2v_lr \
    .join(table_cuisine, prediction_w2v_lr.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

prediction_w2v_lr_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_w2v_lr.csv")
```


```python
lr = LogisticRegression(maxIter=1000, regParam=0.001)
lrModel_tfidf = lr.fit(train_tfidf_ready)
prediction_tfidf_lr = lrModel_tfidf.transform(test_tfidf)

prediction_tfidf_lr.show()
```

    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|        raw_features|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|(20,[0,3,7,8,9,10...|(20,[0,3,7,8,9,10...|[-0.2402417809541...|[0.01878855954716...|       1.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|(20,[0,2,3,6,9,12...|(20,[0,2,3,6,9,12...|[-2.1745024672112...|[4.44780194650880...|       1.0|
    |41580|sausage-links, fe...|[sausage, links, ...|(20,[1,4,11,14,17...|(20,[1,4,11,14,17...|[-0.7961079931199...|[0.01169317278571...|      15.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|(20,[1,3,4,5,7,8,...|(20,[1,3,4,5,7,8,...|[-1.3474940409195...|[0.00153285375896...|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|(20,[1,2,3,4,5,6,...|(20,[1,2,3,4,5,6,...|[0.30701561959254...|[0.02169988623395...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|(20,[0,2,3,6,7,8,...|(20,[0,2,3,6,7,8,...|[1.32447654289936...|[0.01919348003283...|       7.0|
    |19666|grape-juice, oran...|[grape, juice, or...|(20,[6,7,13,17],[...|(20,[6,7,13,17],[...|[1.27498652043362...|[0.05757872900821...|       6.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|(20,[0,2,3,4,5,7,...|(20,[0,2,3,4,5,7,...|[-2.0999875295322...|[0.00165335386512...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|[4.76478999470337...|[0.15882126220460...|       7.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|[-1.6650977416879...|[0.00411949502232...|       3.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|(20,[1,3,4,5,6,7,...|(20,[1,3,4,5,6,7,...|[1.07586298437058...|[0.02912286468371...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|(20,[0,5,6,7,8,9,...|(20,[0,5,6,7,8,9,...|[4.30095978795449...|[0.17386837698370...|       6.0|
    |42969|curry-powder, gro...|[curry, powder, g...|(20,[0,2,3,5,8,9,...|(20,[0,2,3,5,8,9,...|[0.78500920606988...|[0.03514589338205...|       7.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|(20,[1,3,4,6,9,10...|(20,[1,3,4,6,9,10...|[0.92517673760276...|[0.02990206837215...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|(20,[0,3,7,9,12,1...|(20,[0,3,7,9,12,1...|[-0.6255528839358...|[0.01097693826559...|       6.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|(20,[0,3,4,6,8,9,...|(20,[0,3,4,6,8,9,...|[-0.2622930774896...|[0.00705993561766...|       6.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|(20,[1,3,5,6,7,11...|(20,[1,3,5,6,7,11...|[2.07243303648921...|[0.06043941352723...|       6.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|(20,[3,4,7,11,14,...|(20,[3,4,7,11,14,...|[-0.4173006839035...|[0.01655270184305...|       6.0|
    |19001|chopped-green-chi...|[chopped, green, ...|(20,[3,5,6,7,8,9,...|(20,[3,5,6,7,8,9,...|[2.31549868142028...|[0.03961608317532...|       7.0|
    |16526|cold-water, chick...|[cold, water, chi...|(20,[0,3,4,7,8,9,...|(20,[0,3,4,7,8,9,...|[-0.8046307107912...|[0.01349379980824...|      10.0|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
prediction_tfidf_lr = prediction_tfidf_lr \
    .withColumn("prediction", prediction_tfidf_lr["prediction"].cast(IntegerType()))
prediction_tfidf_lr_output = prediction_tfidf_lr \
    .join(table_cuisine, prediction_tfidf_lr.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

prediction_tfidf_lr_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_tfidf_lr")
```

## Decision Tree


```python
DT = DecisionTreeClassifier(maxDepth = 15)
DTmodel = DT.fit(train_w2v)
DTresult = DTmodel.transform(test_w2v)
```


```python
# show detail of the model
```


```python
DTresult.show()
```

    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|[-0.0231250560842...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       6.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|[0.05239258537767...|[0.0,0.0,1.0,0.0,...|[0.0,0.0,0.2,0.0,...|       2.0|
    |41580|sausage-links, fe...|[sausage, links, ...|[-0.0424788616597...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       6.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|[-0.0076890995592...|[0.0,4.0,0.0,0.0,...|[0.0,0.0408163265...|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|[0.00895711896009...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|[-0.0527554646444...|[0.0,63.0,5.0,0.0...|[0.0,0.5338983050...|       1.0|
    |19666|grape-juice, oran...|[grape, juice, or...|[-0.0991512559354...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       5.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|[-0.0522039992614...|[0.0,0.0,4.0,2.0,...|[0.0,0.0,0.046511...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|[0.07571529619218...|[0.0,6.0,0.0,0.0,...|[0.0,0.0202702702...|       7.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|[0.02974281460046...|[3.0,43.0,0.0,0.0...|[0.03529411764705...|       1.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|[-0.0378114992850...|[3.0,0.0,0.0,0.0,...|[0.2,0.0,0.0,0.0,...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|[-0.0036597959096...|[1.0,1.0,0.0,0.0,...|[0.05882352941176...|       6.0|
    |42969|curry-powder, gro...|[curry, powder, g...|[0.01816459600296...|[0.0,0.0,0.0,62.0...|[0.0,0.0,0.0,0.98...|       3.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|[-0.0403843747141...|[15.0,7.0,0.0,0.0...|[0.00720461095100...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|[-0.0262535809539...|[0.0,7.0,0.0,0.0,...|[0.0,0.1,0.0,0.0,...|       6.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|[-0.0219391180202...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|      14.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|[0.02341907822327...|[0.0,7.0,0.0,0.0,...|[0.0,1.0,0.0,0.0,...|       1.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|[-0.0503853579311...|[0.0,29.0,0.0,0.0...|[0.0,0.8285714285...|       1.0|
    |19001|chopped-green-chi...|[chopped, green, ...|[0.07742244796827...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       7.0|
    |16526|cold-water, chick...|[cold, water, chi...|[0.00318516012938...|[0.0,299.0,2.0,0....|[0.0,0.8768328445...|       1.0|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
DTresult.select("probability").take(1)
```




    [Row(probability=DenseVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))]




```python
DTresult = DTresult \
    .withColumn("prediction", DTresult["prediction"].cast(IntegerType()))
DTresult_output = DTresult \
    .join(table_cuisine, DTresult.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

DTresult_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_w2v_dt")
```


```python
DT = DecisionTreeClassifier(maxDepth = 15)
DTmodel = DT.fit(train_tfidf_ready)
DTresult = DTmodel.transform(test_tfidf)

DTresult.show()
```

    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|        raw_features|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|(20,[0,3,7,8,9,10...|(20,[0,3,7,8,9,10...|[0.0,0.0,0.0,2.0,...|[0.0,0.0,0.0,0.18...|       8.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|(20,[0,2,3,6,9,12...|(20,[0,2,3,6,9,12...|[0.0,8.0,0.0,0.0,...|[0.0,0.4,0.0,0.0,...|       1.0|
    |41580|sausage-links, fe...|[sausage, links, ...|(20,[1,4,11,14,17...|(20,[1,4,11,14,17...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       6.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|(20,[1,3,4,5,7,8,...|(20,[1,3,4,5,7,8,...|[0.0,2.0,0.0,0.0,...|[0.0,0.125,0.0,0....|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|(20,[1,2,3,4,5,6,...|(20,[1,2,3,4,5,6,...|[3.0,8.0,1.0,0.0,...|[0.02142857142857...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|(20,[0,2,3,6,7,8,...|(20,[0,2,3,6,7,8,...|[1.0,18.0,0.0,0.0...|[0.04347826086956...|       1.0|
    |19666|grape-juice, oran...|[grape, juice, or...|(20,[6,7,13,17],[...|(20,[6,7,13,17],[...|[1.0,7.0,0.0,1.0,...|[0.01612903225806...|       6.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|(20,[0,2,3,4,5,7,...|(20,[0,2,3,4,5,7,...|[0.0,2.0,0.0,2.0,...|[0.0,0.0246913580...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|      12.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|[1.0,0.0,0.0,0.0,...|[0.2,0.0,0.0,0.0,...|      15.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|(20,[1,3,4,5,6,7,...|(20,[1,3,4,5,6,7,...|[1.0,3.0,1.0,1.0,...|[0.01176470588235...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|(20,[0,5,6,7,8,9,...|(20,[0,5,6,7,8,9,...|[1.0,2.0,0.0,0.0,...|[0.02702702702702...|       7.0|
    |42969|curry-powder, gro...|[curry, powder, g...|(20,[0,2,3,5,8,9,...|(20,[0,2,3,5,8,9,...|[6.0,13.0,0.0,4.0...|[0.08955223880597...|       7.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|(20,[1,3,4,6,9,10...|(20,[1,3,4,6,9,10...|[1.0,1.0,0.0,0.0,...|[0.01086956521739...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|(20,[0,3,7,9,12,1...|(20,[0,3,7,9,12,1...|[2.0,1.0,0.0,4.0,...|[0.09090909090909...|      14.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|(20,[0,3,4,6,8,9,...|(20,[0,3,4,6,8,9,...|[1.0,2.0,0.0,3.0,...|[0.04545454545454...|       6.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|(20,[1,3,5,6,7,11...|(20,[1,3,5,6,7,11...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       6.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|(20,[3,4,7,11,14,...|(20,[3,4,7,11,14,...|[0.0,1.0,0.0,0.0,...|[0.0,0.1111111111...|      14.0|
    |19001|chopped-green-chi...|[chopped, green, ...|(20,[3,5,6,7,8,9,...|(20,[3,5,6,7,8,9,...|[1.0,4.0,0.0,0.0,...|[0.06666666666666...|       1.0|
    |16526|cold-water, chick...|[cold, water, chi...|(20,[0,3,4,7,8,9,...|(20,[0,3,4,7,8,9,...|[7.0,9.0,2.0,2.0,...|[0.04605263157894...|       6.0|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
DTresult = DTresult \
    .withColumn("prediction", DTresult["prediction"].cast(IntegerType()))
DTresult_output = DTresult \
    .join(table_cuisine, DTresult.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

DTresult_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_tfidf_dt")
```

## Random Forest


```python
rf = RandomForestClassifier(maxDepth = 15)
rfmodel = rf.fit(train_w2v)
rfresult = rfmodel.transform(test_w2v)
```


```python
# show detail of the model
```


```python
rfresult.show()
```

    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|[-0.0231250560842...|[0.49775223728141...|[0.02488761186407...|       1.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|[0.05239258537767...|[0.04034767595732...|[0.00201738379786...|       1.0|
    |41580|sausage-links, fe...|[sausage, links, ...|[-0.0424788616597...|[0.22222222222222...|[0.01111111111111...|       6.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|[-0.0076890995592...|[1.00704225352112...|[0.05035211267605...|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|[0.00895711896009...|[1.08794813602467...|[0.05439740680123...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|[-0.0527554646444...|[0.20021166044610...|[0.01001058302230...|       1.0|
    |19666|grape-juice, oran...|[grape, juice, or...|[-0.0991512559354...|[1.43448297856429...|[0.07172414892821...|       7.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|[-0.0522039992614...|[0.0,0.0851449275...|[0.0,0.0042572463...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|[0.07571529619218...|[0.00244260130991...|[1.22130065495810...|       7.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|[0.02974281460046...|[0.38627472974548...|[0.01931373648727...|       1.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|[-0.0378114992850...|[0.22342554644229...|[0.01117127732211...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|[-0.0036597959096...|[2.64946395528132...|[0.13247319776406...|       6.0|
    |42969|curry-powder, gro...|[curry, powder, g...|[0.01816459600296...|[0.02420458004816...|[0.00121022900240...|       3.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|[-0.0403843747141...|[0.16883046733085...|[0.00844152336654...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|[-0.0262535809539...|[0.12728602280811...|[0.00636430114040...|       1.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|[-0.0219391180202...|[0.21666322029225...|[0.01083316101461...|       6.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|[0.02341907822327...|[0.02777777777777...|[0.00138888888888...|       6.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|[-0.0503853579311...|[0.04545454545454...|[0.00227272727272...|       1.0|
    |19001|chopped-green-chi...|[chopped, green, ...|[0.07742244796827...|[0.0,2.0169491525...|[0.0,0.1008474576...|       7.0|
    |16526|cold-water, chick...|[cold, water, chi...|[0.00318516012938...|[0.04784597788791...|[0.00239229889439...|       1.0|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
rfresult.select("probability").take(1)
```




    [Row(probability=DenseVector([0.0249, 0.2861, 0.0143, 0.0103, 0.0254, 0.012, 0.1728, 0.0445, 0.0154, 0.1356, 0.0, 0.0, 0.0141, 0.0033, 0.0623, 0.0209, 0.1387, 0.0, 0.0033, 0.0161]))]




```python
rfresult = rfresult \
    .withColumn("prediction", rfresult["prediction"].cast(IntegerType()))
rfresult_output = rfresult \
    .join(table_cuisine, rfresult.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

rfresult_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_w2v_rf")
```


```python
rf = RandomForestClassifier(maxDepth = 15)
rfmodel = rf.fit(train_tfidf_ready)
rfresult = rfmodel.transform(test_tfidf)

rfresult.show()
```

    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |   id|         ingredients|               words|        raw_features|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    |18009|baking-powder, eg...|[baking, powder, ...|(20,[0,3,7,8,9,10...|(20,[0,3,7,8,9,10...|[0.13311810065630...|[0.00665590503281...|       7.0|
    |28583|sugar, egg-yolks,...|[sugar, egg, yolk...|(20,[0,2,3,6,9,12...|(20,[0,2,3,6,9,12...|[0.25,12.50110300...|[0.0125,0.6250551...|       1.0|
    |41580|sausage-links, fe...|[sausage, links, ...|(20,[1,4,11,14,17...|(20,[1,4,11,14,17...|[0.44111832643592...|[0.02205591632179...|       6.0|
    |29752|meat-cuts, file-p...|[meat, cuts, file...|(20,[1,3,4,5,7,8,...|(20,[1,3,4,5,7,8,...|[0.0,5.0322580645...|[0.0,0.2516129032...|      12.0|
    |35687|ground-black-pepp...|[ground, black, p...|(20,[1,2,3,4,5,6,...|(20,[1,2,3,4,5,6,...|[0.46489139831930...|[0.02324456991596...|       6.0|
    |38527|baking-powder, al...|[baking, powder, ...|(20,[0,2,3,6,7,8,...|(20,[0,2,3,6,7,8,...|[0.19922586183363...|[0.00996129309168...|       1.0|
    |19666|grape-juice, oran...|[grape, juice, or...|(20,[6,7,13,17],[...|(20,[6,7,13,17],[...|[0.75778644077880...|[0.03788932203894...|       6.0|
    |41217|ground-ginger, wh...|[ground, ginger, ...|(20,[0,2,3,4,5,7,...|(20,[0,2,3,4,5,7,...|[0.06818181818181...|[0.00340909090909...|       8.0|
    |28753|diced-onions, tac...|[diced, onions, t...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|[0.10256851278523...|[0.00512842563926...|       7.0|
    |22659|eggs, cherries, d...|[eggs, cherries, ...|(20,[0,3,4,5,6,7,...|(20,[0,3,4,5,6,7,...|[0.29679985201627...|[0.01483999260081...|       3.0|
    |21749|pasta, olive-oil,...|[pasta, olive, oi...|(20,[1,3,4,5,6,7,...|(20,[1,3,4,5,6,7,...|[0.15346888803795...|[0.00767344440189...|       6.0|
    |44967|water, butter, gr...|[water, butter, g...|(20,[0,5,6,7,8,9,...|(20,[0,5,6,7,8,9,...|[2.80481500703369...|[0.14024075035168...|       7.0|
    |42969|curry-powder, gro...|[curry, powder, g...|(20,[0,2,3,5,8,9,...|(20,[0,2,3,5,8,9,...|[0.07725607725607...|[0.00386280386280...|       3.0|
    |44883|pasta, marinara-s...|[pasta, marinara,...|(20,[1,3,4,6,9,10...|(20,[1,3,4,6,9,10...|[0.11596772743617...|[0.00579838637180...|       6.0|
    |20827|salt, custard-pow...|[salt, custard, p...|(20,[0,3,7,9,12,1...|(20,[0,3,7,9,12,1...|[0.28029401154401...|[0.01401470057720...|       6.0|
    |23196|vegetable-oil-coo...|[vegetable, oil, ...|(20,[0,3,4,6,8,9,...|(20,[0,3,4,6,8,9,...|[1.15541310541310...|[0.05777065527065...|       1.0|
    |35387|vanilla-ice-cream...|[vanilla, ice, cr...|(20,[1,3,5,6,7,11...|(20,[1,3,5,6,7,11...|[1.36686442540101...|[0.06834322127005...|       7.0|
    |33780|molasses, hot-sau...|[molasses, hot, s...|(20,[3,4,7,11,14,...|(20,[3,4,7,11,14,...|[0.49752224234957...|[0.02487611211747...|       6.0|
    |19001|chopped-green-chi...|[chopped, green, ...|(20,[3,5,6,7,8,9,...|(20,[3,5,6,7,8,9,...|[0.59782608695652...|[0.02989130434782...|       7.0|
    |16526|cold-water, chick...|[cold, water, chi...|(20,[0,3,4,7,8,9,...|(20,[0,3,4,7,8,9,...|[0.30059291437964...|[0.01502964571898...|       6.0|
    +-----+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
rfresult = rfresult \
    .withColumn("prediction", rfresult["prediction"].cast(IntegerType()))
rfresult_output = rfresult \
    .join(table_cuisine, rfresult.prediction == table_cuisine.label, how='left') \
    .select(["id", "cuisine"])

rfresult_output.coalesce(1).write.option("header","true").csv("gs://big_data_hw/P9120/output_tfidf_rf")
```
