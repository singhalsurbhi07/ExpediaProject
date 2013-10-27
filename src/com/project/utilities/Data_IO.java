package com.project.utilities;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
//import au.com.bytecode.opencsv.CSVReader;

public class Data_IO {
  public static Instances inputs;

  private static Instances csvLoad(String FileType) throws Exception {
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(FileType));
    Instances data = loader.getDataSet();
    return data;
  }

  public static Instances setupTrainFile(String trainData) throws Exception {
    Instances inst = csvLoad(trainData);

    Remove remove = new Remove();
    int[] attr = {0, 1, 2, 7, 14, 51, 52};
    remove.setAttributeIndicesArray(attr);
    remove.setInvertSelection(false);
    remove.setInputFormat(inst);
    Instances instNew = Filter.useFilter(inst, remove);
    instNew.setClassIndex(46);

    // Convert class values to nominal.
    NumericToNominal f = new NumericToNominal();
    f.setInputFormat(instNew);
    f.setAttributeIndices("47");

    instNew = Filter.useFilter(instNew, f);

    System.out.println(instNew.numAttributes());
    System.out.println(instNew.numClasses());
    return instNew;
  }

  public static Instances setupTestFile(String testData) throws Exception {
    Instances inst = csvLoad(testData);

    Remove remove = new Remove();
    int[] attr = {0, 1, 2, 7};
    remove.setAttributeIndicesArray(attr);
    remove.setInvertSelection(false);
    remove.setInputFormat(inst);
    Instances instNew = Filter.useFilter(inst, remove);
    FastVector attributeValues = new FastVector(2);
    attributeValues.addElement("0");
    attributeValues.addElement("1");
    Attribute a = new Attribute("booking_bool", attributeValues);
    instNew.insertAttributeAt(a, 46);
    instNew.setClassIndex(46);
    System.out.println(instNew.numAttributes());
    System.out.println(instNew.numClasses());

    Remove remove1 = new Remove();
    int[] attr1 = {0, 7};
    remove1.setAttributeIndicesArray(attr1);
    remove1.setInvertSelection(true);
    remove1.setInputFormat(inst);
    inputs = Filter.useFilter(inst, remove1);
    return instNew;
  }

  public static Instances readupTestFile(String testData) throws Exception {
    return inputs;
  }
}
