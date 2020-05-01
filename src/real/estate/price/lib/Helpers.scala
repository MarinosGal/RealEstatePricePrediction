package real.estate.price.lib

import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, VectorAssembler}
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.{col, lower, upper}
import org.apache.spark.sql.types.DoubleType

object Helpers {
  def to_lower_case(c: Column): Column = {
    lower(c)
  }
  def to_upper_case(c: Column): Column = {
    upper(c)
  }
  def square(c: Column): Column = {
    c * c
  }
  def transform_columns(df: DataFrame, col_names: Seq[String], fun: Column => Column): DataFrame = {
    df.select(col_names.map(x => fun(col(x))): _*)
  }

  def array_to_double(df: DataFrame, col_names: Seq[String], fun: Column => Column): DataFrame = {
    df.select(col_names.map(x => fun(col(x))): _*)
  }

  def to_float(c: Column): Column = {
    c.cast(DoubleType)
  }

  def cleaning(df: DataFrame): DataFrame = {
    df.drop("date")
  }

  def preProcess(df: DataFrame, input_col: String, output_col: String): (DataFrame, MinMaxScalerModel) = {
    val scaler = new MinMaxScaler().setInputCol(input_col).setOutputCol(output_col)
    val scalerModel = scaler.fit(df)
    val df_normed = scalerModel.transform(df)
    (df_normed, scalerModel)
  }

  def vectorAssembly(df: DataFrame, input_cols: Array[String], output_col: String): DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(input_cols)
      .setOutputCol(output_col)
    assembler.transform(df)
  }
}
