package com.project.classfiers;

import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Instances;

import com.project.classfiers.Utils.Result;
import com.project.utilities.Data_IO;

public class RunClassifier {
    private static Instances readTrainFile(String trainFile) throws Exception {
	Instances trainInst = Data_IO.setupTrainFile(trainFile);
	System.out.println("File Read...");
	return trainInst;
    }

    private static Instances readTestFile(String testFile) throws Exception {
	return Data_IO.setupTestFile(testFile);
    }

    private static Instances getInputForTest(String testFile) throws Exception {
	return Data_IO.readupTestFile(testFile);
    }

    public static void main(String[] args) throws Exception {
	String trainFile = args[0];
	String testFile = args[1];
	String modelFile = args[2];
	String outputFile = args[3];
	String classifierType = args[4];
	String interimFile = args[5];
	String readFromModelFile = args[6];
	ClassifierType cType = ClassifierType.valueOf(classifierType);

	readTrainFile(trainFile);

	Classifier c = null;
	List<Result> result = null;
	switch (cType) {
	case J48:
	    System.out.println("Using J48 classifier");
	    c = J48classfier.train(readTrainFile(trainFile));
	    Utils.saveClassifier(c, modelFile);
	    result = J48classfier.predict(readTestFile(testFile),
		    getInputForTest(testFile), c);
	    break;
	case LogisticRegression:
	    System.out.println("Using Logistic Regression classifier");
	    if (readFromModelFile.equals("no")) {
		c = LogisticRegression.train(readTrainFile(trainFile));
		Utils.saveClassifier(c, modelFile);
	    } else {
		c = Utils.readClassifier(modelFile);
	    }
	    result = LogisticRegression.predict(readTestFile(testFile),
		    getInputForTest(testFile), c);
	    break;
	case MLP:
	    System.out.println("Using MLP classifier");
	    c = MLPImp.train(readTrainFile(trainFile));
	    Utils.saveClassifier(c, modelFile);
	    result = MLPImp.predict(readTestFile(testFile),
		    getInputForTest(testFile), c);
	    break;
	case LinearRegression:
	    System.out.println("Using Logistic Regression classifier");
	    if (readFromModelFile.equals("no")) {
		c = LinearReg.train(readTrainFile(trainFile));
		Utils.saveClassifier(c, modelFile);
	    } else {
		c = Utils.readClassifier(modelFile);
	    }
	    result = LinearReg.predict(readTestFile(testFile),
		    getInputForTest(testFile), c);
	default:
	    break;
	}

	Utils.saveInterimFile(result, interimFile);
	System.out.println("Write Started");
	Utils.writeResultToFile(result, outputFile);
	System.out.println("Done!!!!!!!!");
    }

    public enum ClassifierType {
	J48, LogisticRegression, MLP, LinearRegression
    }
}
