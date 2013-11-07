package com.project.classfiers;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

public class Utils {
  public static class Result {
    public Double srch_id;

    public Double prop_id;

    public Double weight;

    Result() {
    }
  }

  public static void saveClassifier(Classifier c, String model)
      throws Exception {
    SerializationHelper.write(model, c);
    System.out.println("Training finished and model saved!");
  }

  public static Classifier readClassifier(String model) throws Exception {
    return (Classifier) SerializationHelper.read(model);
  }

  public static void saveInterimFile(List<Result> results, String outputFile)
      throws FileNotFoundException {
    File file = new File(outputFile);
    PrintWriter pw = new PrintWriter(file);

    for (Result r : results) {
      pw.println(r.srch_id.intValue() + "," + r.prop_id.intValue()+","+r.weight);
    }
    pw.close();
  }

 
  public static void writeResultToFile(List<Result> results, String outputFile)
      throws FileNotFoundException {
    List<Result> sortedResults = sortResult(results);

    File file = new File(outputFile);
    PrintWriter pw = new PrintWriter(file);

    for (Result r : sortedResults) {
      //System.out.println(r.srch_id.intValue() + "," + r.prop_id.intValue());
      pw.println(r.srch_id.intValue() + "," + r.prop_id.intValue());
    }
    pw.close();
  }

  public static List<Result> sortResult(List<Result> results) {
    Comparator<Result> cmp = new Comparator<Result>() {
      @Override
      public int compare(Result lhs, Result rhs) {
        if (lhs.srch_id == rhs.srch_id) {
          return lhs.weight.compareTo(rhs.weight);
        }
        return lhs.srch_id.compareTo(rhs.srch_id);
      }
    };

    Collections.sort(results, cmp);
    return results;
  }
}
