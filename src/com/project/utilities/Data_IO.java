package com.project.utilities;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.Attribute;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
//import au.com.bytecode.opencsv.CSVReader;

public class Data_IO {
  private static final String TEST_FILE = "data/test_seg.csv";

  private static final String TRAINING_FILE = "data/train_seg.csv";

  // private static List<String[]> readCsvFile(String csvFilename) throws
  // IOException {
  // CSVReader csvReader = new CSVReader(new FileReader(csvFilename));
  // List<String[]> content = csvReader.readAll();
  // csvReader.close();
  // return content;
  // }

  // public static List<String[]> readTestFile() throws IOException{
  // return readCsvFile(TEST_FILE);
  // }
  //
  // public static List<String[]> readTrainFile() throws IOException {
  // return readCsvFile(TRAINING_FILE);
  // }

  private static Instances csvLoad(String FileType) throws Exception {
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(FileType));
    Instances data = loader.getDataSet();
    return data;
  }

  public Instances setupTrainFile(String trainData) throws Exception {
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

    return instNew;
  }

  public Instances setupTestFile(String testData) throws Exception {
    Instances inst = csvLoad(testData);
    Remove remove = new Remove();
    int[] attr = {0, 1, 2, 7};
    remove.setAttributeIndicesArray(attr);
    remove.setInvertSelection(false);
    remove.setInputFormat(inst);
    Instances instNew = Filter.useFilter(inst, remove);
    return instNew;
  }

  public Instances readupTestFile(String testData) throws Exception {
    Instances inst = csvLoad(testData);
    return inst;
  }
}
