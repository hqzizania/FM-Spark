/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.mllib.optimization

import scala.math._

// import breeze.linalg._
import breeze.linalg.{axpy => brzAxpy, norm => brzNorm, Vector => BV}

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * :: DeveloperApi ::
  * A simple updater for gradient descent *without* any regularization.
  * Uses a step-size decreasing with the square root of the number of iterations.
  */
@DeveloperApi
class AdamUpdater extends Serializable {
  private var beta1: Double = 0.9
  private var beta2: Double = 0.999
  private var stepSize: Double = 0.001
  private val epsilon = 1e-8
  private var t: Int = 0

  // take in previors weights, m_, v_, gradient, regVal and return updated weights, m_, v_, regVal
  def compute(
               weightsOld: Vector,
               gradient: Vector,
               mPre: Vector,
               vPre: Vector,
               beta1Power: Double,
               beta2Power: Double,
               iter: Int,
               regParam: Double): (Vector, Vector, Vector, Double, Double, Double) = {
    val mBeta1Power = beta1Power * beta1
    val mBeta2Power = beta2Power * beta2
    val lr: Double = stepSize * sqrt(1 - mBeta2Power) / (1 - mBeta1Power)
    val m: BV[Double] = beta1 * mPre.asBreeze + (1 - beta1) * gradient.asBreeze
    val v: BV[Double] = beta2 * vPre.asBreeze + (1 - beta2) * gradient.asBreeze :* gradient.asBreeze
    val sqrtV: BV[Double] = BV(v.toArray.map(k => math.sqrt(k) + epsilon))
    val adamgrad: BV[Double] = m :/ sqrtV
    val brzWeights: BV[Double] = weightsOld.asBreeze.toDenseVector
    brzAxpy(-lr, adamgrad, brzWeights)

    // finishing
    (Vectors.fromBreeze(brzWeights), Vectors.fromBreeze(m),
      Vectors.fromBreeze(v), mBeta1Power, mBeta2Power, 0)
  }
}