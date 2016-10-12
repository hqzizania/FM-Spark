package org.apache.spark.mllib.regression

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.util.Random

import org.apache.spark.{SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.optimization.{Updater, Gradient}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.Loader._
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.sql.{DataFrame, SQLContext}
import breeze.linalg.{DenseVector => BDV}

/**
  * Factorization Machine model.
  */
class FMModel(val task: Int,
              val factorMatrix: Matrix,
              val weightVector: Option[Vector],
              val intercept: Double,
              val min: Double,
              val max: Double) extends Serializable with Saveable {

  val numFeatures = factorMatrix.numCols
  val numFactors = factorMatrix.numRows

  require(numFeatures > 0 && numFactors > 0)
  require(task == 0 || task == 1)

  def predict(testData: Vector): Double = {
    require(testData.size == numFeatures)

    var pred = intercept
    if (weightVector.isDefined) {
      testData.foreachActive {
        case (i, v) =>
          pred += weightVector.get(i) * v
      }
    }

    for (f <- 0 until numFactors) {
      var sum = 0.0
      var sumSqr = 0.0
      testData.foreachActive {
        case (i, v) =>
          val d = factorMatrix(f, i) * v
          sum += d
          sumSqr += d * d
      }
      pred += (sum * sum - sumSqr) * 0.5
    }

    task match {
      case 0 =>
        Math.min(Math.max(pred, min), max)
      case 1 =>
        1.0 / (1.0 + Math.exp(-pred))
    }
  }

  def predict(testData: RDD[Vector]): RDD[Double] = {
    testData.mapPartitions {
      _.map {
        vec =>
          predict(vec)
      }
    }
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    val data = FMModel.SaveLoadV1_0.Data(factorMatrix, weightVector, intercept, min, max, task)
    FMModel.SaveLoadV1_0.save(sc, path, data)
  }
}

object FMModel extends Loader[FMModel] {

  private object SaveLoadV1_0 {

    def thisFormatVersion = "1.0"

    def thisClassName = "org.apache.spark.mllib.regression.FMModel"

    /** Model data for model import/export */
    case class Data(factorMatrix: Matrix, weightVector: Option[Vector], intercept: Double,
                    min: Double, max: Double, task: Int)

    def save(sc: SparkContext, path: String, data: Data): Unit = {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      // Create JSON metadata.
      val metadata = compact(render(
        ("class" -> this.getClass.getName) ~ ("version" -> thisFormatVersion) ~
          ("numFeatures" -> data.factorMatrix.numCols) ~ ("numFactors" -> data.factorMatrix.numRows)
          ~ ("min" -> data.min) ~ ("max" -> data.max) ~ ("task" -> data.task)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(metadataPath(path))

      // Create Parquet data.
      val dataRDD: DataFrame = sc.parallelize(Seq(data), 1).toDF()
      dataRDD.write.parquet(dataPath(path))
    }

    def load(sc: SparkContext, path: String): FMModel = {
      val sqlContext = new SQLContext(sc)
      // Load Parquet data.
      val dataRDD = sqlContext.parquetFile(dataPath(path))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      checkSchema[Data](dataRDD.schema)
      val dataArray = dataRDD.select("task", "factorMatrix", "weightVector", "intercept", "min", "max").take(1)
      assert(dataArray.length == 1, s"Unable to load FMModel data from: ${dataPath(path)}")
      val data = dataArray(0)
      val task = data.getInt(0)
      val factorMatrix = data.getAs[Matrix](1)
      val weightVector = data.getAs[Option[Vector]](2)
      val intercept = data.getDouble(3)
      val min = data.getDouble(4)
      val max = data.getDouble(5)
      new FMModel(task, factorMatrix, weightVector, intercept, min, max)
    }
  }

  override def load(sc: SparkContext, path: String): FMModel = {
    implicit val formats = DefaultFormats

    val (loadedClassName, version, metadata) = loadMetadata(sc, path)
    val classNameV1_0 = SaveLoadV1_0.thisClassName

    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        val numFeatures = (metadata \ "numFeatures").extract[Int]
        val numFactors = (metadata \ "numFactors").extract[Int]
        val model = SaveLoadV1_0.load(sc, path)
        assert(model.factorMatrix.numCols == numFeatures,
          s"FMModel.load expected $numFeatures features," +
            s" but factorMatrix had columns of size:" +
            s" ${model.factorMatrix.numCols}")
        assert(model.factorMatrix.numRows == numFactors,
          s"FMModel.load expected $numFactors factors," +
            s" but factorMatrix had rows of size:" +
            s" ${model.factorMatrix.numRows}")
        model

      case _ => throw new Exception(
        s"FMModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}


/**
  * :: DeveloperApi ::
  * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
  * For the detailed mathematical derivation, see the reference at
  * http://doi.acm.org/10.1145/2168752.2168771
  */
class FMGradient(val task: Int, val k0: Boolean, val k1: Boolean, val k2: Int,
                 val numFeatures: Int, val min: Double, val max: Double,
                 val r0: Double, val r1: Double, val r2: Double) extends Gradient {

  private def predict(data: Vector, weights: Vector): (Double, Array[Double]) = {

    var pred = if (k0) weights(weights.size - 1) else 0.0

    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          pred += weights(pos + i) * v
      }
    }

    val sum = Array.fill(k2)(0.0)
    for (f <- 0 until k2) {
      var sumSqr = 0.0
      data.foreachActive {
        case (i, v) =>
          val d = weights(i * k2 + f) * v
          sum(f) += d
          sumSqr += d * d
      }
      pred += (sum(f) * sum(f) - sumSqr) * 0.5
    }

    if (task == 0) {
      pred = Math.min(Math.max(pred, min), max)
    }

    (pred, sum)
  }

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    throw new Exception("This part is merged into computeFM()")
  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    throw new Exception("This part is merged into computeFM()")
  }

  def computeFM(data: Vector, label: Double, weights: Vector,
                stepSize: Double, iter: Int, regParam: Double): BDV[Double] = {
    require(data.size == numFeatures)
    val (pred, sum) = predict(data, weights)

    val mult = task match {
      case 0 =>
        pred - label
      case 1 =>
        -label * (1.0 - 1.0 / (1.0 + Math.exp(-label * pred)))
    }

    val thisIterStepSize = stepSize / math.sqrt(iter)
    val len = weights.size
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values

    if (k0) {
      weightsArray(len - 1) = weights(len - 1) - thisIterStepSize * (mult + r0 * weights(len - 1))
    }

    if (k1) {
      val pos = numFeatures * k2
      data.foreachActive {
        case (i, v) =>
          weightsArray(pos + i) -= thisIterStepSize * (v * mult + r1 * weightsArray(pos + i))
      }
    }

    data.foreachActive {
      case (i, v) =>
        val pos = i * k2
        for (f <- 0 until k2) {
          weightsArray(pos + i) -= thisIterStepSize * ((sum(f) * v - weights(pos + f) * v * v) * mult + r2 * weightsArray(pos + i))
        }
    }

    BDV(weightsArray)
  }
}

/**
  *
  */
class FMUpdater() extends Updater {

  override def compute(
               weightsOld: Vector,
               gradient: Vector,
               stepSize: Double,
               iter: Int,
               regParam: Double): (Vector, Double) = {
    throw new Exception("This part is merged into Gradient()")
  }
}
