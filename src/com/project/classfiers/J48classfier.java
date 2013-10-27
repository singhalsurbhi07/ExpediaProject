package com.project.classfiers;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import com.project.classfiers.Utils.Result;

public class J48classfier {
  public static Classifier train(Instances trainInst) throws Exception {
    J48 j48 = new J48();
    // j48.setUnpruned(true);
    j48.buildClassifier(trainInst);
    return j48;
  }

  public static List<Result> predict(Instances testInst, Instances inputs,
      Classifier cl) throws Exception {
    List<Result> reqAttr = new ArrayList<Result>();

    for (int index = 0; index < testInst.numInstances(); index++) {
      Result tempList = new Result();
      Instance newInstance = testInst.instance(index);
      Instance input = inputs.instance(index);

      tempList.srch_id = input.value(0);
      tempList.prop_id = input.value(1);

      double[] dist = cl.distributionForInstance(newInstance);
      tempList.weight = getProbabilityOfBooking(dist[0], dist[1]);
      reqAttr.add(tempList);
    }
    return reqAttr;
  }

  private static double getProbabilityOfBooking(double class1Wt, double class2Wt) {
    return (class2Wt / (class1Wt + class2Wt)) * 1E4;
  }
}
