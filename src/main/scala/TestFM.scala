import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD



object TestFM extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTFM").setMaster("local[4]"))

    //customer dataset format convertor
    /*
    val rawData = sc.textFile("data/5line", 4).map(_.split("\\s")).map(x => {
      if (x(0).toInt > 3)
        x(0) = "1"
      else
        x(0) = "-1"
      val v: Array[(Int, Double)] = x.drop(1).map(_.split(":"))
        .map(x => (x(0).toInt - 1, x(1).toDouble))
        .sortBy(_._1)
      (x(0).toInt, v)
    }).repartition(4)

    val length = rawData.map(_._2.last._1).max + 1

    val training: RDD[LabeledPoint] = rawData.map{case(label, v) => LabeledPoint(label, Vectors.sparse(length, v.map(_._1), v.map(_._2)))}

    */

    val training = MLUtils.loadLibSVMFile(sc, "data/a9a").repartition(4)

    val testing = MLUtils.loadLibSVMFile(sc, "data/a9a.t")


    val fm1: FMModel = FMWithSGD.train(training, task = 1, numIterations
      = 5, stepSize = 0.01, dim = (true, true, 8), regParam = (0, 0.0, 0.01), initStd = 0.01)

    val scores: RDD[(Int, Int)] = fm1.predict(testing.map(_.features)).map(x => if(x >= 0.5) 1 else -1).zip(testing.map(_.label.toInt))
    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()

    println(s"accuracy = $accuracy")
    sc.stop()
  }
}
