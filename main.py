from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (StringIndexer, OneHotEncoder, RegexTokenizer, StopWordsRemover,
                                CountVectorizer, IDF, Imputer, VectorAssembler, IndexToString)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

SAMPLE_FRACTION = 0.03         # ~45K rows (0.1 = ~150K is much slower on a laptop)
TUNE = False                   # True = 3-fold CV over regParam (about 10x slower)
# Strict version: ONLY the 5 required features (category, subscription_type, issue_description,
# previous_tickets, time_since_creation). The diagnostic showed no other column adds anything.
EXTRA_CAT = []
EXTRA_NUM = []
USE_CLASS_WEIGHTS = False      # True = the model starts predicting rare classes (Urgent), accuracy drops a bit
RUN_DIAGNOSTIC = True          # prints how much each column predicts priority (held-out accuracy)

spark = (SparkSession.builder.appName("CustomerSupportPriority")
         .master("local[4]").config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ---------- TASK 1: load + missing values ----------
cat_cols = ["category", "subscription_type"] + EXTRA_CAT
num_cols = ["previous_tickets", "time_since_creation"] + EXTRA_NUM

df = spark.read.parquet("0000.parquet").sample(False, SAMPLE_FRACTION, 42)
df = df.filter(F.col("priority").isNotNull() & (F.trim("priority") != ""))
df = df.withColumn("priority", F.initcap(F.trim("priority")))
df = df.fillna({**{c: "Unknown" for c in cat_cols}, "issue_description": ""})
df = df.dropDuplicates(["ticket_id"])

df = df.withColumn("created_ts", F.col("created_at").cast("timestamp"))
ref = df.agg(F.max("created_ts")).first()[0]
df = df.withColumn("time_since_creation",
                   (F.unix_timestamp(F.lit(ref)) - F.unix_timestamp("created_ts")) / 3600)

data = df.select(*cat_cols, "issue_description", *num_cols, "priority").cache()
data.count()   # materialise so every later step sees the SAME rows and the SAME train/test split
train, test = data.randomSplit([0.8, 0.2], seed=42)
print("train:", train.count(), "test:", test.count())
print("features -> categorical:", cat_cols, "| numeric:", num_cols)

# ---------- TASK 2: encode + text features ----------
idx = [c + "_idx" for c in cat_cols]
ohe = [c + "_ohe" for c in cat_cols]
num_imp = [c + "_imp" for c in num_cols]
pipeline = Pipeline(stages=[
    StringIndexer(inputCols=cat_cols, outputCols=idx, handleInvalid="keep"),
    OneHotEncoder(inputCols=idx, outputCols=ohe, handleInvalid="keep"),
    RegexTokenizer(inputCol="issue_description", outputCol="words", pattern="\\W+"),
    StopWordsRemover(inputCol="words", outputCol="filtered_words"),
    CountVectorizer(inputCol="filtered_words", outputCol="raw_features", vocabSize=5000, minDF=5),
    IDF(inputCol="raw_features", outputCol="text_features"),
    Imputer(inputCols=num_cols, outputCols=num_imp, strategy="median"),
    VectorAssembler(inputCols=ohe + ["text_features"] + num_imp, outputCol="features"),
    StringIndexer(inputCol="priority", outputCol="label"),
])
pipeline_model = pipeline.fit(train)
train_data = pipeline_model.transform(train)
test_data = pipeline_model.transform(test)

lr_kwargs = dict(featuresCol="features", labelCol="label", maxIter=100)
if USE_CLASS_WEIGHTS:
    cnt = {r["label"]: r["count"] for r in train_data.groupBy("label").count().collect()}
    tot = sum(cnt.values())
    wmap = F.create_map(*[x for k, v in cnt.items() for x in (F.lit(float(k)), F.lit(tot / (len(cnt) * v)))])
    train_data = train_data.withColumn("w", wmap[F.col("label")])
    lr_kwargs["weightCol"] = "w"
train_data = train_data.cache()

# ---------- TASK 3: Logistic Regression ----------
if TUNE:
    lr = LogisticRegression(**lr_kwargs)
    grid = ParamGridBuilder().addGrid(lr.regParam, [1e-4, 1e-3, 1e-2]).build()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid,
                        evaluator=MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy"),
                        numFolds=3, seed=42)
    lr_model = cv.fit(train_data).bestModel
else:
    lr_model = LogisticRegression(regParam=1e-3, **lr_kwargs).fit(train_data)
print("best regParam:", lr_model.getRegParam())

# ---------- TASK 4 + 5: evaluation and predictions ----------
pred = lr_model.transform(test_data)
labels = pipeline_model.stages[-1].labels
final = IndexToString(inputCol="prediction", outputCol="predicted_priority", labels=labels).transform(pred).cache()

print("\n" + "=" * 45 + "\n MODEL PERFORMANCE (test set)\n" + "=" * 45)
for name, m in [("Accuracy", "accuracy"), ("Precision (weighted)", "weightedPrecision"),
                ("Recall (weighted)", "weightedRecall"), ("F1 (weighted)", "f1")]:
    v = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=m).evaluate(pred)
    print(f" {name:22s}: {v:.4f}  ({v * 100:.2f}%)")

total = final.count()
correct = final.filter(F.col("priority") == F.col("predicted_priority")).count()
print(f"\n Correct predictions: {correct:,} of {total:,} test tickets")

print("\nConfusion matrix (rows = actual, columns = predicted):")
(final.groupBy("priority").pivot("predicted_priority", ["Low", "Medium", "High", "Urgent"])
      .count().fillna(0).orderBy("priority").show())

print("Correct rate per actual priority:")
(final.groupBy("priority")
      .agg(F.count("*").alias("tickets"),
           F.round(F.avg((F.col("priority") == F.col("predicted_priority")).cast("double")), 4).alias("correct_rate"))
      .orderBy("priority").show())

print("Sample predictions (actual vs predicted):")
final.select("priority", "predicted_priority").show(20, truncate=False)

# ---------- DIAGNOSTIC: how much does each column predict priority? ----------
if RUN_DIAGNOSTIC:
    tr, te = df.randomSplit([0.8, 0.2], 42)
    n_te = te.count()
    base = te.groupBy("priority").count().agg(F.max("count")).first()[0] / n_te

    def lookup_acc(keys):
        w = Window.partitionBy(*keys).orderBy(F.desc("count"))
        best = (tr.groupBy(*keys, "priority").count().withColumn("r", F.row_number().over(w))
                  .filter("r = 1").select(*keys, F.col("priority").alias("guess")))
        return te.join(best, keys, "left").filter(F.col("priority") == F.col("guess")).count() / n_te

    print(f"\nHELD-OUT ACCURACY of 'most common priority for each value' (always-majority baseline = {base:.3f}):")
    for keys in (["category"], ["issue_type_id"], ["issue_complexity_score"], ["subscription_type"],
                 ["customer_segment"], ["previous_tickets"], ["channel"], ["product"], ["region"], ["language"],
                 ["issue_type_id", "issue_complexity_score"], ["issue_type_id", "subscription_type"],
                 ["issue_type_id", "customer_segment"], ["issue_type_id", "previous_tickets"]):
        print(f"  {str(keys):52s} {lookup_acc(keys):.3f}")

# ---------- EXPORT FOR POWER BI ----------
import os
import pandas as pd

os.makedirs("powerbi_data", exist_ok=True)
ORDER = {"Low": 1, "Medium": 2, "High": 3, "Urgent": 4}

# 1) test-set predictions (the rows the model never trained on)
pdf = (final.select("category", "subscription_type", "previous_tickets", "time_since_creation",
                    F.col("priority").alias("actual_priority"), "predicted_priority")
            .toPandas())
pdf["is_correct"] = (pdf["actual_priority"] == pdf["predicted_priority"]).astype(int)
pdf["actual_order"] = pdf["actual_priority"].map(ORDER)
pdf["predicted_order"] = pdf["predicted_priority"].map(ORDER)
pdf.to_csv("powerbi_data/test_predictions.csv", index=False)

# 2) actual vs predicted counts per priority (zero rows kept, e.g. Urgent never predicted)
def counts(col, kind):
    return (pdf[col].value_counts().reindex(list(ORDER), fill_value=0)
              .rename_axis("priority").reset_index(name="tickets").assign(type=kind))
dist = pd.concat([counts("actual_priority", "Actual"), counts("predicted_priority", "Predicted")])
dist["order"] = dist["priority"].map(ORDER)
dist.to_csv("powerbi_data/priority_distribution.csv", index=False)

# 3) the cleaned ~45K-row sample the model was built from
data.toPandas().to_csv("powerbi_data/cleaned_sample.csv", index=False)
print("Saved to", os.path.abspath("powerbi_data"))