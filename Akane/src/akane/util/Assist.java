package akane.util;

import akane.entity.TimePoint;
import akane.entity.TimeSeries;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Time;
import java.util.ArrayList;


import org.nd4j.linalg.indexing.NDArrayIndex;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.clusterers.SimpleKMeans;
import weka.core.SelectedTag;

import java.util.HashMap;
import java.util.Map;
import java.util.*;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

public class Assist {
  public static String PATH = "./Akane/ExpData/";

  public TimeSeries readData(String filename, int index, String splitOp, int tslen) {
    TimeSeries timeSeries = new TimeSeries();

    try {
      FileReader fr = new FileReader(PATH + filename);
      BufferedReader br = new BufferedReader(fr);

      String line;
      long timestamp;
      double value;
      TimePoint tp;

      int linenum = 0;
      while ((line = br.readLine()) != null) {
        String[] vals = line.split(splitOp);
        timestamp = Long.parseLong(vals[0]);
        value = Double.parseDouble(vals[index]);

        tp = new TimePoint(timestamp, value);
        timeSeries.addTimePoint(tp);
        linenum++;
        if(linenum >= tslen)
          break;
      }

      br.close();
      fr.close();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }

    return timeSeries;
  }
  /**
   * RMS sqrt(|modify - truth|^2 / len)
   *
   * @param truthSeries truth
   * @param resultSeries after repair
   * @return RMS error
   */
  public double calcRMS(TimeSeries truthSeries, TimeSeries resultSeries) {
    double cost = 0;
    double delta;
    int len = truthSeries.getLength();

    for (int i = 0; i < len; ++i) {
      delta = resultSeries.getTimeseries().get(i).getModify()
          - truthSeries.getTimeseries().get(i).getValue();

      cost += delta * delta;
    }
    cost /= len;

    return Math.sqrt(cost);
  }

  public double[] TS2List(TimeSeries timeseries){
    ArrayList<TimePoint> totalList = timeseries.getTimeseries();
    int n = totalList.size();

    double[] values = new double[n];

    for(int i = 0; i < n; i++){
      values[i] = totalList.get(i).getValue();
    }

    return values;
  }

  public TimeSeries List2TS(double[] values){
    int n = values.length;

    TimeSeries resultSeries = new TimeSeries();
    for(int i = 0; i < n; i++){
      TimePoint tp = new TimePoint(i+1, values[i]);
      resultSeries.addTimePoint(tp);
    }

    return resultSeries;
  }

  public static class Triplet implements Comparable<Triplet> {
    private int first;
    private String second;
    private Double third;

    public Triplet(int first, String second, Double third) {
      this.first = first;
      this.second = second;
      this.third = third;
    }

    public int getFirst() {return first;}
    public void setFirst(int first) {this.first = first;}
    public String getSecond() {return second;}
    public void setSecond(String second) {this.second = second;}
    public Double getThird() {return third;}
    public void setThird(Double third) {this.third = third;}

    @Override
    public int compareTo(Triplet other) {
      return this.third.compareTo(other.getThird());
    }

    @Override
    public String toString() {
      return "(" + first + ", " + second + ", " + third + ")";
    }
  }

  public void heapify(Triplet arr[], int N, int i, int[] posRecord)
  {
    int largest = i; // Initialize largest as root
    int l = 2 * i + 1; // left = 2*i + 1
    int r = 2 * i + 2; // right = 2*i + 2

    // If left child is larger than root
    if (l < N && arr[l].compareTo(arr[largest]) > 0)
      largest = l;

    // If right child is larger than largest so far
    if (r < N && arr[r].compareTo(arr[largest]) > 0)
      largest = r;

    // If largest is not root
    if (largest != i) {
      Triplet swap = arr[i];
      arr[i] = arr[largest];
      arr[largest] = swap;

      int tempPos = posRecord[arr[i].getFirst()];
      posRecord[arr[i].getFirst()] = posRecord[arr[largest].getFirst()];
      posRecord[arr[largest].getFirst()] = tempPos;

      // Recursively heapify the affected sub-tree
      heapify(arr, N, largest, posRecord);
    }
  }

  // Function to build a Max-Heap from the Array
  public void buildHeap(Triplet[] arr, int[] posRecord, int N)
  {
    // Index of last non-leaf node
    int startIdx = (N / 2) - 1;

    // Perform reverse level order traversal
    // from last non-leaf node and heapify
    // each node
    for (int i = startIdx; i >= 0; i--) {
      heapify(arr, N, i, posRecord);
    }
  }

  public void replaceHeap(Triplet[] arr, int[] posRecord, int index, Triplet sub, int N) {
    int targetIndex = posRecord[index];
    // Step 2: Replace the target element with the given Triplet sub
    arr[targetIndex] = sub;

    // Step 3: Reheapify the heap after replacement
    // Check and adjust upwards
    while(targetIndex != 0 && arr[targetIndex].compareTo(arr[(targetIndex - 1) / 2]) > 0) {
      // Swap current node with its parent
      Triplet temp = arr[targetIndex];
      arr[targetIndex] = arr[(targetIndex - 1) / 2];
      arr[(targetIndex - 1) / 2] = temp;

      int tempPos = posRecord[arr[targetIndex].getFirst()];
      posRecord[arr[targetIndex].getFirst()] = posRecord[arr[(targetIndex - 1) / 2].getFirst()];
      posRecord[arr[(targetIndex - 1) / 2].getFirst()] = tempPos;

      // Move to the parent index
      targetIndex = (targetIndex - 1) / 2;
    }

    // Check and adjust downwards
    heapify(arr, N, targetIndex, posRecord);
  }

  // A utility function to print the array
  // representation of Heap
  public void printHeap(Triplet arr[], int N)
  {
    System.out.println(
        "Array representation of Heap is:");

    for (int i = 0; i < N; ++i)
      System.out.println(arr[i].toString());

    System.out.println();
  }
}
