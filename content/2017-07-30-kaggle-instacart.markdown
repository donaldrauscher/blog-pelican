Title: Using Google Dataproc for the Kaggle Instacart Challenge
Date: 2017-07-30
Tags: spark, dataproc, gcp, kaggle
Slug: kaggle-instacart

I recently competed in [this Kaggle competition](https://www.kaggle.com/c/instacart-market-basket-analysis). It's a challenging problem because we're not just trying to predict whether someone will buy a specific product; we're trying to predict the *entirety* of someone's next order.  And there are 49,688 possible products.  Furthermore, in the train orders, 60% of the products being ordered are reorders and 40% of the products are being ordered for the first time.  Predicting which products will be reordered is MUCH easier than predicting which products will be ordered for the first time.  

For this challenge, I built my models in Spark, specifically PySpark on Google Dataproc.  I used [this initialization script](https://github.com/GoogleCloudPlatform/dataproc-initialization-actions/tree/master/jupyter) to install Jupyter Notebook on the cluster's master node.  I used BigQuery to handle a lot of the feature engineering.  The [Dataproc BigQuery Connector](https://cloud.google.com/dataproc/docs/concepts/connectors/bigquery) isn't great; for starters, it doesn't allow you to execute queries.  So I used the GCP client lib to built my tables, and the BigQuery Connector to export for Dataproc.

Note: My model focuses only on reorders, which has the previously noted deficiency of only addressing ~60% of the problem! I tried building a collaborative filtering model to predict new product trial, but it performed poorly.  As a next step, I'd like to try building a recommendation model that recommends new products for a specific order, e.g. FPGrowth or Word2Vec.



```python
import numpy as np
import pandas as pd

from google.cloud import bigquery

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```

### Create BQ table with data for model


```python
bq_client = bigquery.Client()
job_config = bigquery.QueryJobConfig()

table_ref = bq_client.dataset('instacart').table('reorder_model')
job_config.destination = table_ref
job_config.write_disposition = 'WRITE_TRUNCATE'

query = """
    WITH users AS (
      SELECT user_id, COUNT(*) AS num_orders, SUM(days_since_prior_order) AS days_bw_first_last_order
      FROM instacart.orders
      WHERE eval_set = "prior"
      GROUP BY 1
    ), user_product AS (
      SELECT orders.user_id, op.product_id,
        COUNT(*) AS num_orders, SUM(op.reordered) AS num_reorders,
        MIN(orders.order_number) AS first_order_number, MIN(days_since_first_order) AS first_order_day,
        MAX(orders.order_number) AS last_order_number, MAX(days_since_first_order) AS last_order_day,
        AVG(op.add_to_cart_order) AS avg_cart_order
      FROM instacart.order_products__prior AS op
      INNER JOIN (
        SELECT *,
          SUM(COALESCE(days_since_prior_order,0)) OVER (PARTITION BY user_id ORDER BY order_number ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS `days_since_first_order`
        FROM instacart.orders
        WHERE eval_set = "prior"
      ) AS orders USING(order_id)
      GROUP BY 1,2
    ), user_product_features AS (
      SELECT up.user_id, up.product_id,
        up.num_orders / users.num_orders AS perc_all_orders,
        SAFE_DIVIDE(up.num_reorders, users.num_orders - up.first_order_number) AS perc_reorder,
        SAFE_DIVIDE(up.num_orders, users.days_bw_first_last_order) AS orders_per_day,
        SAFE_DIVIDE(up.num_reorders, users.days_bw_first_last_order - up.first_order_day) AS reorders_per_day,
        up.first_order_number, up.first_order_day, up.last_order_number, up.last_order_day, up.avg_cart_order,
        users.days_bw_first_last_order
      FROM user_product AS up
      INNER JOIN users AS users USING(user_id)
    ), user_features AS (
      SELECT orders.user_id,
        ANY_VALUE(users.num_orders) AS num_orders,
        ANY_VALUE(users.days_bw_first_last_order) AS days_bw_first_last_order,
        ANY_VALUE(users.days_bw_first_last_order) / ANY_VALUE(users.num_orders) AS avg_days_bw_orders,
        COUNT(*) / ANY_VALUE(users.num_orders) AS num_products_per_order,
        SUM(op.reordered) / SUM(CASE WHEN orders.order_number > 1 THEN 1 ELSE 0 END) AS perc_reorder,
        COUNT(DISTINCT op.product_id) AS num_products,
        COUNT(DISTINCT products.aisle_id) AS num_aisles,
        COUNT(DISTINCT products.department_id) AS num_departments
      FROM instacart.orders AS orders
      INNER JOIN instacart.order_products__prior AS op USING(order_id)
      INNER JOIN instacart.products AS products USING(product_id)
      INNER JOIN users USING(user_id)
      GROUP BY 1
    ), product_features AS (
      SELECT product_id, aisle_id, department_id,
        num_users / num_users_tot AS perc_users,
        num_orders / num_orders_tot AS perc_all_orders,
        num_reorder / num_reorder_tot AS perc_reorder
      FROM (
        SELECT products.product_id, products.aisle_id, products.department_id,
          COUNT(DISTINCT orders.user_id) AS num_users,
          COUNT(*) AS num_orders,
          SUM(op.reordered) AS num_reorder
        FROM instacart.orders AS orders
        INNER JOIN instacart.order_products__prior AS op USING(order_id)
        INNER JOIN instacart.products AS products USING(product_id)
        GROUP BY 1,2,3
      ) AS x
      INNER JOIN (
        SELECT COUNT(DISTINCT user_id) AS num_users_tot,
          COUNT(*) AS num_orders_tot,
          SUM(CASE WHEN order_number > 1 THEN 1 ELSE 0 END) AS num_reorder_tot
        FROM instacart.orders
        WHERE eval_set = "prior"
      ) AS y ON 1=1
    ), all_features AS (
      SELECT
        upf.user_id,
        upf.product_id,
        pf.aisle_id,
        pf.department_id,
        upf.perc_all_orders AS upf_perc_all_orders,
        upf.perc_reorder AS upf_perc_reorder,
        upf.orders_per_day AS upf_orders_per_day,
        upf.reorders_per_day AS upf_reorders_per_day,
        upf.first_order_number AS upf_first_order_number,
        upf.first_order_day AS upf_first_order_day,
        upf.last_order_number AS upf_last_order_number,
        upf.last_order_day AS upf_last_order_day,
        upf.avg_cart_order AS upf_avg_cart_order,
        uf.num_orders AS uf_num_orders,
        uf.num_products_per_order AS uf_num_products_per_order,
        uf.perc_reorder AS uf_perc_reorder,
        uf.days_bw_first_last_order AS uf_days_bw_first_last_order,
        uf.avg_days_bw_orders AS uf_avg_days_bw_orders,
        uf.num_products AS uf_num_products,
        uf.num_aisles AS uf_num_aisles,
        uf.num_departments AS uf_num_departments,
        pf.perc_users AS pf_perc_users,
        pf.perc_all_orders AS pf_perc_all_orders,
        pf.perc_reorder AS pf_perc_reorder
      FROM user_product_features AS upf
      INNER JOIN user_features AS uf USING(user_id)
      INNER JOIN product_features AS pf USING(product_id)
    )
    SELECT af.*,
      # a few other features that need to computed based on order
      af.uf_days_bw_first_last_order - af.upf_last_order_day + o.days_since_prior_order AS upf_days_since_last_order,
      o.order_number - af.upf_last_order_number AS upf_orders_since_last_order,
      # train vs. test and reordered (only for train)
      o.eval_set,
      o.order_id,
      CASE WHEN o.eval_set = "test" THEN NULL ELSE LEAST(COALESCE(op_train.order_id,0),1) END AS reordered
    FROM all_features AS af
    INNER JOIN instacart.orders AS o ON af.user_id = o.user_id AND o.eval_set IN ('train','test')
    LEFT JOIN instacart.order_products__train AS op_train ON o.order_id = op_train.order_id AND af.product_id = op_train.product_id
"""

query_job = bq_client.query(query, job_config=job_config)
result = query_job.result(timeout=600)
assert query_job.state == 'DONE'
```

### Pull data from BQ into Spark DF


```python
# for deleting temp files when we're done
def cleanup(sess, input_directory):
    input_path = sess._jvm.org.apache.hadoop.fs.Path(input_directory)
    input_path.getFileSystem(sess._jsc.hadoopConfiguration()).delete(input_path, True)
```


```python
# set up spark session
sess = SparkSession.builder\
    .appName("Model builder")\
    .config("spark.executor.cores", 2)\
    .config("spark.executor.memory", "7g")\
    .config("spark.network.timeout", 3000)\
    .config("spark.shuffle.io.maxRetries", 10)\
    .getOrCreate()

bucket = sess._sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = sess._sc._jsc.hadoopConfiguration().get('fs.gs.project.id')
input_dir = 'gs://{}/hadoop/tmp/bigquery/pyspark_input'.format(bucket)
output = 'gs://instacart-data/outputs/reorder_test_pred.csv'
```


```python
# load data from bq
conf = {
    'mapred.bq.project.id': project,
    'mapred.bq.gcs.bucket': bucket,
    'mapred.bq.temp.gcs.path': input_dir,
    'mapred.bq.input.project.id': project,
    'mapred.bq.input.dataset.id': 'instacart',
    'mapred.bq.input.table.id': 'reorder_model',
}

cleanup(sess, input_dir)

data_raw = sess._sc.newAPIHadoopRDD(
    'com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat',
    'org.apache.hadoop.io.LongWritable',
    'com.google.gson.JsonObject',
    conf=conf)

data_json = data_raw.map(lambda x: x[1])
data_df = sess.read.json(data_json).repartition(sess._sc.defaultParallelism*2)
```


```python
# cast integers
data_df = data_df\
    .withColumn('label', col("reordered").cast('integer'))\
    .withColumn('aisle_id', col("aisle_id").cast('integer'))\
    .withColumn('department_id', col("department_id").cast('integer'))\
    .withColumn('user_id', col("user_id").cast('integer'))\
    .withColumn('product_id', col("product_id").cast('integer'))\
    .withColumn('order_id', col("order_id").cast('integer'))\
    .withColumn('uf_num_orders', col("uf_num_orders").cast('integer'))\
    .withColumn('uf_days_bw_first_last_order', col("uf_days_bw_first_last_order").cast('integer'))\
    .withColumn('uf_num_aisles', col("uf_num_aisles").cast('integer'))\
    .withColumn('uf_num_departments', col("uf_num_departments").cast('integer'))\
    .withColumn('uf_num_products', col("uf_num_products").cast('integer'))\
    .withColumn('upf_first_order_day', col("upf_first_order_day").cast('integer'))\
    .withColumn('upf_first_order_number', col("upf_first_order_number").cast('integer'))\
    .withColumn('upf_last_order_day', col("upf_last_order_day").cast('integer'))\
    .withColumn('upf_last_order_number', col("upf_last_order_number").cast('integer'))\
    .withColumn('upf_orders_since_last_order', col("upf_orders_since_last_order").cast('integer'))\
    .withColumn('upf_days_since_last_order', col("upf_days_since_last_order").cast('integer'))
```

### Train/test split and set up ML pipeline


```python
# split into train/test
train = data_df.filter(data_df.eval_set == 'train').cache()
test = data_df.filter(data_df.eval_set == 'test').cache()

train_user, validate_user = train.select('user_id').distinct().randomSplit([0.8, 0.2], seed=1)

train2 = train.join(broadcast(train_user), 'user_id').cache()
validate = train.join(broadcast(validate_user), 'user_id').cache()
```


```python
# construct pipeline
xvar1 = ["upf_perc_all_orders", "upf_perc_reorder", "upf_orders_per_day", "upf_reorders_per_day", \
         "upf_first_order_number", "upf_first_order_day", "upf_last_order_number", "upf_last_order_day", \
         "upf_avg_cart_order", "upf_days_since_last_order", "upf_orders_since_last_order"]

xvar2 = ["uf_num_orders", "uf_num_products_per_order", "uf_perc_reorder", \
         "uf_days_bw_first_last_order", "uf_avg_days_bw_orders", "uf_num_products", "uf_num_aisles", \
         "uf_num_departments"]

xvar3 = ["pf_perc_users", "pf_perc_all_orders", "pf_perc_reorder"]

xvar4 = ["aisle_id", "department_id"]

xvar = xvar1 + xvar2 + xvar3 + xvar4

null_counts = train.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in train.columns))\
                .toPandas().transpose()
null_col = list(null_counts.index[null_counts[0].nonzero()])

imp = Imputer(strategy="median", inputCols=null_col, outputCols=null_col)
va = VectorAssembler(inputCols=xvar, outputCol="features")
gb = GBTClassifier(seed=0, maxIter=10)
pipeline = Pipeline(stages=[imp, va, gb])
```

### Hyperparameter tuning


```python
# hyperparameter tuning
param_grid = ParamGridBuilder()\
    .addGrid(gb.minInstancesPerNode, [10, 25])\
    .addGrid(gb.maxDepth, [5, 7])\
    .addGrid(gb.stepSize, [0.1, 0.2])\
    .build()

eva = BinaryClassificationEvaluator(metricName='areaUnderROC')
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=param_grid,
                    numFolds=3,
                    evaluator=eva)

cv_model = cv.fit(train2)

best_func = np.argmax if eva.isLargerBetter() else np.argmin
best_idx = best_func(cv_model.avgMetrics)
best_score = cv_model.avgMetrics[best_idx]
best_param = param_grid[best_idx]

print("Best CV score: {}".format(best_score))
print("Best CV param: {}".format(best_param))

```

<pre>Best CV score: 0.8279589610834616
Best CV param: {Param(parent='GBTClassifier_40d0a7396b9c4171e238', name='minInstancesPerNode', doc='Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.'): 25, Param(parent='GBTClassifier_40d0a7396b9c4171e238', name='maxDepth', doc='Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'): 7, Param(parent='GBTClassifier_40d0a7396b9c4171e238', name='stepSize', doc='Step size to be used for each iteration of optimization (>= 0).'): 0.2}</pre>

### Determine cutoff which maximizes mean F1 score

This model focuses on predicting reorders of products ordered previously.  We need to include products ordered for the first time, for which we are not generating predictions, in the recall denominator.  Otherwise, we will get a F1 estimate that is too high.


```python
# proportion of products that are reorders
query = "SELECT AVG(reordered) FROM instacart.order_products__train"
query_job = bq_client.query(query)
prop_reordered = list(query_job.result())[0][0]
```


```python
# calculate probabilities for validation set
true_prob = udf(lambda x: float(x[-1]))

validate_pred = cv_model.transform(validate, best_param)
validate_pred = validate_pred.select(true_prob('probability').alias('probability').cast('float'), 'label')

validate_pred = validate_pred.withColumn("probability_bkt", round(col("probability"), 2))
validate_pred_df = validate_pred.groupBy("probability_bkt")\
                                .agg(sum('label').alias('sum'), count('label').alias('count'))\
                                .toPandas()
```


```python
# calculate precision/recall at different thresholds
def precision_fn(df, cutoff):
    x = df.loc[df.probability_bkt >= cutoff, ['sum','count']].apply(np.sum)
    return x[0] / x[1]

def recall_fn(df, cutoff):
    relevant = np.sum(df['sum']) / prop_reordered
    return np.sum(df['sum'][df.probability_bkt >= cutoff]) / relevant

thresholds = np.arange(0.01, 0.99, 0.01)
precision = np.array([precision_fn(validate_pred_df, x) for x in thresholds])
recall = np.array([recall_fn(validate_pred_df, x) for x in thresholds])
f1 = (2*precision*recall)/(precision+recall)
optimal_threshold = thresholds[np.nanargmax(f1)]

print("Optimal threshold: {}".format(optimal_threshold))
print("Optimal threshold F1: {}".format(np.nanmax(f1)))
```

<pre>Optimal threshold: 0.16
Optimal threshold F1: 0.3407145351781795</pre>

### Generate predictions for test set


```python
# tune model on entire data
model = pipeline.fit(train, best_param)
```


```python
# create predictions for test set
collapse = udf(lambda x: ' '.join([str(i) for i in x]))

test_order = test.select("order_id").distinct()

test_pred = model.transform(test)
true_prob = udf(lambda x: float(x[-1]))

test_pred = test_pred.select('order_id', 'product_id', true_prob('probability').alias('probability').cast('float'))\
                     .filter(col("probability") >= optimal_threshold)\
                     .groupBy('order_id').agg(collect_list('product_id').alias('products'))
test_pred = test_pred.withColumn('products', collapse('products'))
test_pred = test_order.join(test_pred, on='order_id', how='left')
```


```python
# export
cleanup(sess, output)
test_pred.repartition(1).write.option('header', 'true').csv(output)
```


```python
# cleanup
cleanup(sess, input_dir)
```

====

My final submission had a F1 score of 0.37.  You can find all of my code for this project [here](https://github.com/donaldrauscher/kaggle-instacart).  Cheers!
