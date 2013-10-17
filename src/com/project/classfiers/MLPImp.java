package com.project.classfiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

import com.project.utilities.Data_IO;

import weka.core.Attribute;
public class MLPImp {
	/**
	 * @throws Exception
	 */
	private static String MODEL_FILE="/home/surbhi/Downloads/CMPE239/Project/modelFile.txt";

	private static void  train() throws Exception{
		Data_IO dataGen=new Data_IO();
		Instances trainInst=dataGen.setupTrainFile();
		MultilayerPerceptron MLPNet =  new MultilayerPerceptron(); // create a WEKA MLP structure
		String[] options = weka.core.Utils.splitOptions("-H 100,100");
		MLPNet.setOptions(options);
		MLPNet.buildClassifier(trainInst) ; // build an MLP classifier on the training set

		SerializationHelper.write(MODEL_FILE, MLPNet);
		//Evaluation eval = new Evaluation(trainInst);
		//eval.evaluateModel(MLPNet, trainInst);
		//System.out.println(eval.correct());
		System.out.println("Training finished and model saved!");
	}

	private static List<Result> predict() throws Exception {
		Classifier cl = (Classifier) SerializationHelper.read(MODEL_FILE);
		Data_IO dataGen=new Data_IO();

		Attribute a = new Attribute("booking_bool");
		Instances testInst = dataGen.setupTrainFile();
		//testInst.insertAttributeAt(a, 46);
		
		Instances testInstWithoutChopping = dataGen.readupTestFile();
		
		ArrayList<Result> reqAttr=new ArrayList<Result>();
		
		for(int index=0;index<testInst.numInstances();index++){
			Result tempList=new Result();
			Instance newInstance = testInst.instance(index);
			
			Instance newInstanceWithoutChopping = testInstWithoutChopping.instance(index);
			tempList.srch_id = newInstanceWithoutChopping.value(0);
			tempList.prop_id = newInstanceWithoutChopping.value(7);

			double[] dist = cl.distributionForInstance(newInstance);
			tempList.weight = getProbabilityOfBooking(dist[0], dist[1]);
			reqAttr.add(tempList);
		}
		
		return reqAttr;
		
		/*
		for(ArrayList<Double>d:reqAttr){
			for(double val:d){
				System.out.print(val+", ");
			}
			System.out.println();
		}*/
	}
	
	private static double getProbabilityOfBooking(double class1Wt, double class2Wt){
		return (class2Wt/(class1Wt+class2Wt))*1E4;
	}
	
	private static List<Result> sortResult(List<Result> results){
		Comparator<Result> cmp = new Comparator<Result>() {
			  @Override
		      public int compare(Result lhs, Result rhs) {
		        if(lhs.srch_id == rhs.srch_id){
		        	return lhs.weight.compareTo(rhs.weight);
		        } 
		        return lhs.srch_id.compareTo(rhs.srch_id);
		      }
		};
		
		Collections.sort(results, cmp);
		return results;
	}

	public static void main(String [] args) throws Exception{
		train();
		List<Result> results  = predict();
		List<Result> sortedResults = sortResult(results);
		for(Result r: sortedResults){
			System.out.println(r.srch_id+","+r.prop_id);
		}
	}
	
	static class Result {
		public Double srch_id;
		
		public Double prop_id;
		
		public Double weight;
		
		Result(){}
	}	
}
