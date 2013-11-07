package com.project.utilities;

import java.io.File;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

//import au.com.bytecode.opencsv.CSVReader;

public class Data_IO {
    public static Instances inputs;

    private static Instances csvLoad(String FileType) throws Exception {
	CSVLoader loader = new CSVLoader();
	loader.setOptions(weka.core.Utils.splitOptions("-M NULL"));
	loader.setSource(new File(FileType));
	Instances data = loader.getDataSet();
	return data;
    }

    public static Instances setupTrainFile(String trainData) throws Exception {
	Instances inst = csvLoad(trainData);

	Remove remove = new Remove();
	int[] attr = { 0, 1, 2, 7, 14, 52, 53 };
	remove.setAttributeIndicesArray(attr);
	remove.setInvertSelection(false);
	remove.setInputFormat(inst);
	inst = Filter.useFilter(inst, remove);

	// Convert class values to nominal.
	NumericToNominal f = new NumericToNominal();
	f.setOptions(weka.core.Utils
		.splitOptions("-R 1,4,5,6,7,12,19,22,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45"));
	// f.setOptions(weka.core.Utils.splitOptions("-R 1,2,5,8,13,14,20,24,25,27,28,30,31,33,34,36,37,39,40,42,43,45,46,48"));
	f.setInputFormat(inst);
	inst = Filter.useFilter(inst, f);

	inst.setClassIndex(46);

	ReplaceMissingValues rm = new ReplaceMissingValues();
	// rm.setOptions(weka.core.Utils.splitOptions("-unset-class-temporarily"));
	rm.setInputFormat(inst);
	inst = Filter.useFilter(inst, rm);

	System.out.println(inst.numAttributes());
	System.out.println(inst.numClasses());

	for (int i = 0; i < inst.numInstances(); i++) {
	    if (inst.instance(i).value(inst.numAttributes() - 1) == 1) {
		inst.instance(i).setWeight(50);
		System.out.println("Instance value:" + inst.instance(1));
	    }
	}

	for (int i = 0; i < inst.numInstances(); i++) {
	    System.out.print(inst.instance(i).value(inst.numAttributes() - 1) + "--");
	    System.out.println(inst.instance(i).weight());
	}

	return inst;
    }

    public static Instances setupTestFile(String testData) throws Exception {
	Instances inst = csvLoad(testData);

	Remove remove1 = new Remove();
	int[] attr1 = { 0, 7 };
	remove1.setAttributeIndicesArray(attr1);
	remove1.setInvertSelection(true);
	remove1.setInputFormat(inst);
	inputs = Filter.useFilter(inst, remove1);

	Remove remove = new Remove();
	int[] attr = { 0, 1, 2, 7 };
	remove.setAttributeIndicesArray(attr);
	remove.setInvertSelection(false);
	remove.setInputFormat(inst);
	inst = Filter.useFilter(inst, remove);

	// Convert class values to nominal.
	NumericToNominal f = new NumericToNominal();
	f.setOptions(weka.core.Utils
		.splitOptions("-R 1,4,5,6,7,12,19,22,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45"));
	f.setInputFormat(inst);
	inst = Filter.useFilter(inst, f);

	// Replace Missing Values
	ReplaceMissingValues rm = new ReplaceMissingValues();
	// rm.setOptions(weka.core.Utils.splitOptions("-unset-class-temporarily"));
	rm.setInputFormat(inst);
	inst = Filter.useFilter(inst, rm);

	// Add class attribute
	// List<String> attributeValues = new ArrayList<String>(2);
	// attributeValues.add("0");
	// attributeValues.add("1");
	// Attribute a = new Attribute("booking_bool", attributeValues);
	Attribute a = new Attribute("booking_bool");
	inst.insertAttributeAt(a, 46);
	inst.setClassIndex(46);

	// for(int i = 0;i<48;i++) {
	// System.out.print(inst.attribute(i).type()+"--");
	// }

	System.out.println();
	System.out.println(inst.numAttributes());
	// System.out.println(inst.numClasses());

	return inst;
    }

    public static Instances readupTestFile(String testData) throws Exception {
	return inputs;
    }
}
