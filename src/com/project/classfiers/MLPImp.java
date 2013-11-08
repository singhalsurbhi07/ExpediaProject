package com.project.classfiers;


import java.util.ArrayList;
import java.util.List;


import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;

import weka.core.Instance;
import weka.core.Instances;

import com.project.classfiers.Utils.Result;

public class MLPImp {

  public static Classifier train(Instances trainInst) throws Exception {
    MultilayerPerceptron MLPNet = new MultilayerPerceptron(); // create a WEKA
    String[] options = weka.core.Utils.splitOptions("-H 100,100");
    MLPNet.setOptions(options);
    MLPNet.buildClassifier(trainInst); // build an MLP classifier on the
    System.out.println("Classifier Trained...");
    return MLPNet;
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

      try {
      	tempList.weight = cl.distributionForInstance(newInstance)[0];
      } catch (Exception e) {
	tempList.weight = 0.0; 
      }
      reqAttr.add(tempList);
    }

    return reqAttr;
  }

  private static double getProbabilityOfBooking(double class1Wt, double class2Wt) {
    return (class2Wt / (class1Wt + class2Wt)) * 1E4;
  }

}
