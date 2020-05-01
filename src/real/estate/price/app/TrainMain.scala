package real.estate.price.app

import org.apache.log4j._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.functions.{max, min}
import org.apache.spark.sql.{DataFrame, SparkSession}
import real.estate.price.lib.Helpers._

object TrainMain {

  def main(args: Array[String]): Unit = {
    val logger = Logger.getLogger("org")
    logger.setLevel(Level.ERROR)

    val applyCv = if (args.isEmpty) false else args(0).toBoolean

    println("Initialize RealEstatePricePrediction application")
    val spark = SparkSession.builder().master("local[*]").appName("RealEstatePricePrediction").getOrCreate()

    val df = spark.read.option("header", true)
      .option("inferSchema", true)
      .csv("./data/kc_house_data.csv").cache()

    println("Preprocessing...")
    val df_clean = cleaning(df) // drop unnecessary columns

    // convert features to vector assembly
    val df_features = vectorAssembly(df_clean.drop("price").withColumnRenamed("id", "features_id"), df_clean.drop("price", "id").columns, "features")
    val df_label = df_clean.select("price", "id").withColumnRenamed("id", "label_id") //vectorAssembly(df_clean.select("price","id").withColumnRenamed("id","label_id"), Array("price"), "label")

    // normalize feature vector with zero to one MinMaxScaler
    val (df_features_normed, minMaxScalerModelFeatures) = preProcess(df_features, "features", "features_normed")

    // normalize labels with zero to one MinMaxScaler
    val min_price = df_label.select(min("price")).first().getDecimal(0)
    val max_price = df_label.select(max("price")).first().getDecimal(0)
    val df_label_normed = df_label.withColumn("label", (df_label("price") - min_price) / max_price.subtract(min_price))

    // join features and label for splitting dataframe into train and test set
    val df_normed = df_features_normed.join(df_label_normed, df_features_normed("features_id") === df_label_normed("label_id"), "inner")
    val splits = df_normed.randomSplit(Array(0.8, 0.2))
    val (train_df, test_df) = (splits(0), splits(1))

    // apply Cross Validation
    if (applyCv) {
      println("Cross Validation...")
      val gbt = new GBTRegressor()

      val paramGrid = new ParamGridBuilder()
        .addGrid(gbt.maxIter, Array(10,20,30))
        .build()

      val cv = new CrossValidator()
        .setEstimator(gbt)
        .setEvaluator(new RegressionEvaluator())
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3)
        .setParallelism(2)

      // Run train validation split, and choose the best set of parameters.
      val model = cv.fit(train_df)

      val test_predictions = model.transform(test_df)
      val train_predictions = model.transform(train_df)

      evaluate(test_predictions, train_predictions)

    } else {
      println("Gradient Boosted Regressor")
      val gbt = new GBTRegressor()
        .setLabelCol("label")
        .setFeaturesCol("features_normed")
        .setMaxIter(30)
        .fit(train_df)

      val test_predictions = gbt.transform(test_df)
      val train_predictions = gbt.transform(train_df)

      evaluate(test_predictions, train_predictions)
    }

    def evaluate(test_predictions: DataFrame, train_predictions: DataFrame) = {
      val evaluator = new RegressionEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("rmse")

      val test_rmse = evaluator.evaluate(test_predictions)
      println("test rmse = "+test_rmse)
      val train_rmse = evaluator.evaluate(train_predictions)
      println("train rmse = "+train_rmse)
    }

  }

}
