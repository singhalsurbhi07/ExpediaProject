package com.project.classfiers;

import com.project.utilities.Data_IO;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;

public class J48classfier {
	public J48classfier() throws Exception {
		// TODO Auto-generated constructor stub
		//runClassifier();
	}
	
	private static void runClassifier() throws Exception{
		Data_IO dataGen=new Data_IO();
		Instances data= dataGen.setupFileForJ48classifier();
		String[] options = new String[1];
		options[0] = "-U"; 
		LinearRegression tree = new LinearRegression(); 
		tree.setOptions(options); 
		tree.buildClassifier(data); 
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(tree, data);
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	}
	
	public static void main(String [] args) throws Exception{
		runClassifier();
	}
}
