package com.project.classfiers;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

import com.project.classfiers.Utils.Result;

public class LinearReg {
    public static Classifier train(Instances trainInst) throws Exception {
	LinearRegression linearReg = new LinearRegression();
	// logisticReg.setOptions(weka.core.Utils.splitOptions("-N -F 1"));
	linearReg.buildClassifier(trainInst);
	return linearReg;
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
	    // System.out.println(cl.classifyInstance(newInstance));
	    try {
		double[] dist = cl.distributionForInstance(newInstance);
		System.out.println(dist[0]);
		tempList.weight = dist[0];// getProbabilityOfBooking(dist[0],
					  // dist[1]);
	    } catch (Exception e) {
		tempList.weight = 0.0;// getProbabilityOfBooking(dist[0],
				      // dist[1]);
	    }
	    reqAttr.add(tempList);
	}
	return reqAttr;
    }

    private static double getProbabilityOfBooking(double class1Wt, double class2Wt) {
	return (class2Wt / (class1Wt + class2Wt)) * 1E4;
    }
}
