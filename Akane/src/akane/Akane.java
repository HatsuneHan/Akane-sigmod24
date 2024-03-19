package akane;
import akane.util.Assist;
import java.util.Arrays;

import akane.entity.TimeSeries;

import java.util.ArrayList;

import java.util.HashSet;

import org.apache.commons.math3.linear.SingularMatrixException;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.clusterers.SimpleKMeans;
import weka.core.SelectedTag;

import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

public class Akane {

  private TimeSeries timeseries;
  private int K; // markov chain order
  private int[] Budgetlist; // diffeorent budget to clean
  private int R;

  public Akane(TimeSeries timeSeries, int k, int[] budgetlist){
    setTimeseries(timeSeries);
    setK(k);
    setBudgetList(budgetlist);
    setR(-1);
  }

  public Akane(TimeSeries timeSeries, int k, int[] budgetlist, int r){
    setTimeseries(timeSeries);
    setK(k);
    setBudgetList(budgetlist);
    setR(r);
  }

  public Akane(TimeSeries timeSeries, int k){
    setTimeseries(timeSeries);
    setK(k);
    setBudgetList(null);
    setR(-1);
  }

  public void setTimeseries(TimeSeries timeSeries){this.timeseries = timeSeries;}

  public void setK(int k){this.K = k;}

  public void setR(int r){this.R = r;}

  public void setBudgetList(int[] budgetlist){this.Budgetlist = budgetlist;}

  public double[] clusterData(double[] data, int numClusters) {
    try {
      // 1. Convert data to Instances format
      ArrayList<Attribute> attributes = new ArrayList<>();
      attributes.add(new Attribute("value"));

      Instances instances = new Instances("Dataset", attributes, data.length);

      for (double value : data) {
        DenseInstance instance = new DenseInstance(1);
        instance.setValue(attributes.get(0), value);
        instances.add(instance);
      }

      // 2. Cluster data using SimpleKMeans
      SimpleKMeans kMeans = new SimpleKMeans();

      kMeans.setInitializationMethod(new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION));
      kMeans.setNumClusters(numClusters);
      kMeans.buildClusterer(instances);

      // 3. Print clustering results
      Instances c = kMeans.getClusterCentroids();
      double[] centroids = new double[c.numInstances()];
      for (int i = 0; i < c.numInstances(); i++) {
        centroids[i] = c.instance(i).value(0);
      }

      return centroids;

    } catch (Exception e) {
      e.printStackTrace();
      return null;
    }
  }

  public double[] getBreakpoints(double[] centroids, double minValue, double maxValue){
    int cn = centroids.length;
    double[] breakpoints = new double[cn+1];
    breakpoints[0] = minValue;
    for(int i = 0; i < cn-1; i++){
      breakpoints[i+1] = (centroids[i]+centroids[i+1])/2;
    }
    breakpoints[cn] = maxValue;
    return breakpoints;
  }

  public String[] symbolizeData(double[] data, double[] breakpoints) {
    int n = data.length;
    int bn = breakpoints.length;
    String[] symbols = new String[n];

    for(int i = 0; i < n; i++){
      for(int j = 0; j < bn-1; j++){
        if(breakpoints[j] <= data[i] && data[i] <= breakpoints[j+1]){
          symbols[i] = Integer.toString(j);
        }
      }
    }

    return symbols;
  }

  public double computeDaviesBouldinIndex(double[] data, String[] symbols, double[] centroids) {
    int numClusters = centroids.length;
    double[] Si = new double[numClusters]; // scatter for each cluster
    int[] Ti = new int[numClusters];

    // 1. Compute scatter (Si) for each cluster
    for (int i = 0; i < data.length; i++) {
      int clusterId = Integer.parseInt(symbols[i]); // assuming symbols contain cluster ids
      Si[clusterId] += Math.pow(data[i] - centroids[clusterId], 2);
      Ti[clusterId] += 1;
    }

    for (int i = 0; i < numClusters; i++) {
      // Average and square root to get scatter (Si)
      Si[i] = Math.sqrt(Si[i] / Ti[i]);
    }

    // 2. Compute separation (Rij) and the Davies-Bouldin score
    double[] RijMax = new double[numClusters];
    for (int i = 0; i < numClusters; i++) {
      for (int j = 0; j < numClusters; j++) {
        if (i != j) {
          double Mij = Math.abs(centroids[i] - centroids[j]);
          double Rij = (Si[i] + Si[j]) / Mij;
          RijMax[i] = Math.max(RijMax[i], Rij);
        }
      }
    }

    // 3. Compute the final Davies-Bouldin score
    double DBIndex = 0.0;
    for (int i = 0; i < numClusters; i++) {
      DBIndex += RijMax[i];
    }
    DBIndex /= numClusters;

    return DBIndex;
  }

  public String[] getSymbolSet(int r){
    String[] idxlist = new String[r];
    for(int i = 0; i < r; i++){
      idxlist[i] = Integer.toString(i);
    }
    return idxlist;
  }

  public static Map<ArrayList<String>, Integer> getCount(String[] words, int k) {
    Map<ArrayList<String>, Integer> countMap = new HashMap<>();

    for(int p = 1; p <= k; p++){
      for (int i = 0; i <= words.length - p; i++) {
        ArrayList<String> gram = new ArrayList<String>();
        for (int j = 0; j < p; j++) {
          gram.add(words[i+j]);
        }
        if (countMap.containsKey(gram)) {
          int curVal = countMap.get(gram);
          countMap.put(gram, curVal + 1);
        } else {
          countMap.put(gram, 1);
        }
      }
    }

    return countMap;
  }

  public static Map<ArrayList<String>, ArrayList<Integer>> getPos(String[] words, int k) {
    Map<ArrayList<String>, ArrayList<Integer>> posMap = new HashMap<>();

    for (int i = 0; i <= words.length - k; i++) {
      ArrayList<String> gram = new ArrayList<>();
      for (int j = 0; j < k; j++) {
        gram.add(words[i+j]);
      }

      ArrayList<Integer> curPos;
      if (posMap.containsKey(gram)) {
        curPos = posMap.get(gram);
      } else {
        curPos = new ArrayList<>();
      }
      curPos.add(i);
      posMap.put(gram, curPos);
    }

    return posMap;
  }

  public static int getDist(String symbol, String target){
    if(symbol.equals(target)){
      return 0;
    }else{
      return 1;
    }
  }

  public static double getScore(ArrayList<String> gram,
                                Map<ArrayList<String>, Integer> countMap,
                                int r, int n){
    int cnt_front = 0;
    int cnt_all = 0;
    int gramsize = gram.size();
    if(gramsize == 1){
      cnt_front = n;
      if(countMap.containsKey(gram)){
        cnt_all = countMap.get(gram);
      }
    }else{
      ArrayList<String> gram_front = new ArrayList<>();
      for(int i = 0; i < gramsize-1; i++){
        gram_front.add(gram.get(i));
      }
      if(countMap.containsKey(gram_front)){
        cnt_front = countMap.get(gram_front);
      }
      if(countMap.containsKey(gram)){
        cnt_all = countMap.get(gram);
      }
    }

    double probability = (cnt_all*1.0+1.0)/(cnt_front*1.0+r*1.0);
    return Math.log(probability);
//    return probability;
  }

  public static void recurInitial(INDArray D, int curk, int k, String[] symbols,
                                  int r, Map<ArrayList<String>, Integer> countMap,
                                  int n, String[] idxlist,
                                  ArrayList<String> chrlist, int budget){

    if(curk == 1){
      for(String chr : idxlist){
        chrlist.add(chr);

        int cost = 0;
        double score = 0;

        for(int i = 0; i < k; i++){
          cost += getDist(symbols[k-i-1], chrlist.get(i));

          ArrayList<String> tmplist = new ArrayList<>();
          for(int j = 0; j <= i; j++){
            tmplist.add(chrlist.get(k-1-j));
          }
          score += getScore(tmplist, countMap, r, n);
        }

        int[] d_index = new int[k+2];
        d_index[0] = k-1;

        for(int i = 2; i < k+2; i++){
          d_index[i] = Integer.parseInt(chrlist.get(k+1-i));
        }

        for(int w = 0; w < budget+1; w++){
          d_index[1] = w;
          if(w >= cost){
            D.putScalar(d_index, score);
            //            System.out.println(score);
          }
        }

        chrlist.remove(chrlist.size()-1);
      }
      return;
    }else{
      for(String chr : idxlist){
        chrlist.add(chr);
        recurInitial(D, curk-1, k, symbols, r, countMap, n, idxlist, chrlist, budget);
        chrlist.remove(chrlist.size()-1);
      }
    }
  }

  public static void recurDP(INDArray D, INDArray record, int curi, int curk,
                             int k, String[] symbols, int r,
                             Map<ArrayList<String>, Integer> countMap, int n,
                             String[] idxlist, ArrayList<String> chrlist,
                             int w, int prevw){
    if(curk == k){
      for(String chr : idxlist){
        chrlist.add(chr);
        int cost = getDist(symbols[curi], chr);
        if(cost > w){
          chrlist.remove(chrlist.size()-1);
          continue;
        }
        recurDP(D, record, curi, curk-1, k, symbols,
            r, countMap, n, idxlist, chrlist, w, w-cost);
        chrlist.remove(chrlist.size()-1);
      }
    }else if(curk == 0){
      for(String chr : idxlist){
        chrlist.add(chr);
        int[] d_index_prev = new int[k+2];
        d_index_prev[0] = curi-1;
        d_index_prev[1] = prevw;
        for(int i = 2; i < k+2; i++){
          d_index_prev[i] = Integer.parseInt(chrlist.get(k+2-i));
        }
        double prev_val = D.getDouble(d_index_prev);
        if(prev_val == Double.NEGATIVE_INFINITY){
          chrlist.remove(chrlist.size()-1);
          continue;
        }

        ArrayList<String> tmplist = new ArrayList<>();
        for(int i = 0; i < k+1; i++){
          tmplist.add(chrlist.get(k-i));
        }

        double score = getScore(tmplist, countMap, r, n);
        double accuml = score + prev_val;

        int[] d_index = new int[k+2];
        d_index[0] = curi;
        d_index[1] = w;
        for(int i = 2; i < k+2; i++){
          d_index[i] = Integer.parseInt(chrlist.get(k+1-i));
        }

        if(D.getDouble(d_index) < accuml){
          D.putScalar(d_index, accuml);
          record.putScalar(d_index, Integer.parseInt(chr));
        }

        chrlist.remove(chrlist.size()-1);
      }

    }else{
      for(String chr : idxlist){
        chrlist.add(chr);
        recurDP(D, record, curi, curk-1, k, symbols, r,
            countMap, n, idxlist, chrlist, w, prevw);
        chrlist.remove(chrlist.size()-1);
      }
    }
  }

  public static void recurTraceInitial(INDArray D, int budget, int curk, int k,
                                       ArrayList<String> chrlist, String[] idxlist,
                                       int n, double[] bsf, String[] bsflist,
                                       int[] targetW){
    if(curk == 1){
      for(String chr : idxlist){
        chrlist.add(chr);
        for(int w = 0; w <= budget; w++){
          int[] d_index = new int[k+2];
          d_index[0] = n-1;
          d_index[1] = w;
          for(int i = 2; i < k+2; i++){
            d_index[i] = Integer.parseInt(chrlist.get(k+1-i));
          }

          double curval = D.getDouble(d_index);

          if(curval > bsf[0]){
            bsf[0] = curval;
            targetW[0] = w;
            for(int i = 0; i < k; i++){
              bsflist[i] = chrlist.get(i);
            }
          }
        }
        chrlist.remove(chrlist.size()-1);
      }
    }else{
      for(String chr : idxlist){
        chrlist.add(chr);
        recurTraceInitial(D, budget, curk-1, k, chrlist, idxlist, n, bsf, bsflist, targetW);
        chrlist.remove(chrlist.size()-1);
      }
    }
  }

  public static String[] recurTrace(String[] symbols, INDArray D, INDArray record,
                                    int k, int r, int n, int budget, String[] idxlist){
    String[] results = new String[n];
    double[] bsf = {Double.NEGATIVE_INFINITY};
    String[] bsflist = new String[k];
    int[] targetW = {-1};

    ArrayList<String> chrlist = new ArrayList<>();
    recurTraceInitial(D, budget, k, k, chrlist, idxlist, n, bsf, bsflist, targetW);

    for(int i = 0; i < k; i++){
      results[n-i-1] = bsflist[i];
    }

    int j = n-k;

    int[] d_index = new int[k+2];
    d_index[0] = n-1;
    d_index[1] = targetW[0];
    for(int i = 2; i < k+2; i++){
      d_index[i] = Integer.parseInt(bsflist[k+1-i]);
    }

    String targetS = Integer.toString(record.getInt(d_index));
    while(j>1){
      results[j-1] = targetS;
      int tmpW = getDist(symbols[j+k-1], bsflist[0]);
      targetW[0] = targetW[0] - tmpW;

      for(int i = 0; i < k-1; i++){
        bsflist[i] = bsflist[i+1];
      }
      bsflist[k-1] = targetS;

      d_index[0] = j+k-2;
      d_index[1] = targetW[0];
      for(int i = 2; i < k+2; i++){
        d_index[i] = Integer.parseInt(bsflist[k+1-i]);
      }

      targetS = Integer.toString(record.getInt(d_index));
      j--;
    }
    results[0] = targetS;
    return results;
  }

  public static double[] cleanDataCentroids(double data[], String[] symbols,
                                            String[] results, double[] centroids, int n){
    double[] cleandata = new double[n];

    for(int i = 0; i < n; i++){
      if(symbols[i].equals(results[i])){
        cleandata[i] = data[i];
      }else{
        cleandata[i] = centroids[Integer.parseInt(results[i])];
      }
    }

    return cleandata;
  }

  public static double getLRVal(int lb, int rb, int curi, int[] diff, int[] sumdiff,
                                Map<ArrayList<String>, Integer> countMap, String[] results,
                                Map<ArrayList<String>, ArrayList<Integer>> posMap, int k,
                                double defaultVal, double data[]) {
    int max_pstart = -1;
    int max_times = 0;

    for(int i = lb; i <= rb; i++){
      ArrayList<String> pattern = new ArrayList<>();
      for (int j = i; j <= i+k; j++) {
        pattern.add(results[j]);
      }
      int times = countMap.getOrDefault(pattern, 0);
      int diffnum = sumdiff[i+k+1]-sumdiff[i];

      if(times > max_times && diffnum <= (k+1.0)/2.0){
        max_pstart = i;
        max_times = times;
      }
    }
    if(max_pstart == -1){
      return defaultVal;
    }

    int y_idx = curi - max_pstart;
    ArrayList<Integer> x_idx = new ArrayList<>();

    for(int i = max_pstart; i <= max_pstart + k; i++){
      if(diff[i] == 0 && i != curi){
        x_idx.add(i-max_pstart);
      }
    }

    ArrayList<String> pattern = new ArrayList<>();
    for (int i = max_pstart; i <= max_pstart+k; i++) {
      pattern.add(results[i]);
    }

    ArrayList<Integer> poslist = posMap.get(pattern);
    ArrayList<Integer> legalposlist = new ArrayList<>();

    for (int pos : poslist) {
      boolean flag = false;
      for (int offset : x_idx){
        if(diff[pos+offset] == 1){
          flag = true;
          break;
        }
      }
      if(diff[pos+y_idx] == 1){
        flag = true;
      }
      if(!flag){
        legalposlist.add(pos);
      }
    }

    int legallen = legalposlist.size();
    int xlen = x_idx.size();

    double[][] X = new double[legallen][xlen];
    double[] Y = new double[legallen];

    int actualLen = 0;
    HashSet<String> uniqueRows = new HashSet<>();

    for(int i = 0; i < legallen; i++){

      double[] tempRow = new double[xlen];
      for(int j = 0; j < xlen; j++){
        tempRow[j] = data[legalposlist.get(i)+x_idx.get(j)];
      }

      String rowStr = java.util.Arrays.toString(tempRow);
      if(!uniqueRows.contains(rowStr)){ 
        uniqueRows.add(rowStr);
        X[actualLen] = tempRow;
        Y[actualLen] = data[legalposlist.get(i)+y_idx];
        actualLen++;
      }
    }
    if(actualLen <= xlen)
      return defaultVal;
    X = java.util.Arrays.copyOf(X, actualLen);
    Y = java.util.Arrays.copyOf(Y, actualLen);

    try {
      OLSMultipleLinearRegression ols = new OLSMultipleLinearRegression();
      ols.newSampleData(Y, X);
      double[] params = ols.estimateRegressionParameters();
      double cleanLRData = params[0];
      for (int i = 0; i < xlen; i++) {
        cleanLRData += params[i + 1] * data[max_pstart + x_idx.get(i)];
      }
      return cleanLRData;
    }catch (SingularMatrixException e){
      return defaultVal;
    }
  }

  public static double[] cleanDataLR(double data[], String[] symbols,
                                     String[] results, double[] breakpoints,
                                     double[] centroids, int k, int n,
                                     Map<ArrayList<String>, Integer> countMap,
                                     Map<ArrayList<String>, ArrayList<Integer>> posMap) {
    double[] cleandata = new double[n];

    int[] diff = new int[n];
    for(int i = 0; i < n; i++){
      if(results[i].equals(symbols[i])){
        diff[i] = 0;
      }else{
        diff[i] = 1;
      }
    }

    int[] sumdiff = new int[n+1];
    sumdiff[0] = 0;
    for(int i = 0; i < n; i++){
      sumdiff[i+1] = sumdiff[i] + diff[i];
    }

    for(int i = 0; i < n; i++){
      if(symbols[i].equals(results[i])){
        cleandata[i] = data[i];
      }else{
        int idx = Integer.parseInt(results[i]);
        int lb = Math.max(i-k,0);
        int rb = Math.min(i, n-k-1);
        double cleanLRVal = getLRVal(lb, rb, i, diff, sumdiff, countMap,
            results, posMap, k, centroids[idx], data);
        if(breakpoints[idx] <= cleanLRVal && cleanLRVal <= breakpoints[idx+1]){
          cleandata[i] = cleanLRVal;
        }else if(cleanLRVal < breakpoints[idx]){
          cleandata[i] = breakpoints[idx];
        }else{
          cleandata[i] = breakpoints[idx+1];
        }
      }
    }
    return cleandata;
  }

  public ArrayList<TimeSeries> mainAkane(){
    if(Budgetlist == null){
      System.out.println("Please provide budgetlist to clean.");
      return null;
    }
    Assist assist = new Assist();
    double[] data = assist.TS2List(timeseries);
    int n = data.length;

    double maxValue = Arrays.stream(data).max().getAsDouble();
    double minValue = Arrays.stream(data).min().getAsDouble();

    double minresult = Double.POSITIVE_INFINITY;
    int bestR = -1;

    if(R == -1){
      for(int testr = 2; testr < 11; testr++){
        double[] centroids = clusterData(data, testr);
        Arrays.sort(centroids);
        double[] breakpoints = getBreakpoints(centroids, minValue, maxValue);
        String[] symbols = symbolizeData(data, breakpoints);
        double result = computeDaviesBouldinIndex(data, symbols, centroids);
//      System.out.println(testr + " " + result);
        if(minresult > result){
          minresult = result;
          bestR = testr;
        }
      }
      System.out.println("Choose Calculated Cluster Number R = " + bestR);
    }else{
      bestR = R;
      System.out.println("Choose Given Cluster Number R = " + bestR);
    }


    double[] centroids = clusterData(data, bestR);
    Arrays.sort(centroids);
    double[] breakpoints = getBreakpoints(centroids, minValue, maxValue);
    String[] symbols = symbolizeData(data, breakpoints);
    String[] idxlist = getSymbolSet(bestR);

    Map<ArrayList<String>, Integer> countMap = getCount(symbols,K+1);
    Map<ArrayList<String>, ArrayList<Integer>> posMap = getPos(symbols, K+1);

    int maxBudget = Arrays.stream(Budgetlist).max().getAsInt();

    int[] d_shape = new int[K+2];
    d_shape[0] = n;
    d_shape[1] = maxBudget+1;
    for(int i = 2; i < K+2; i++){
      d_shape[i] = bestR;
    }
    INDArray D = Nd4j.zeros(d_shape).subi(Double.POSITIVE_INFINITY);
    INDArray record = Nd4j.zeros(d_shape);
    ArrayList<String> chrlist = new ArrayList<>();
    recurInitial(D, K, K, symbols, bestR, countMap, n, idxlist, chrlist, maxBudget);

    for(int w = 0; w < maxBudget+1; w++){
//      if(w % 10 == 0){
//        System.out.println(w);
//      }
      for(int i = K; i < n; i++){
        recurDP(D, record, i, K, K, symbols, bestR,
            countMap, n, idxlist, chrlist, w, -1); // any value for prevw is ok here
      }
    }

    ArrayList<TimeSeries> resultSeriesList = new ArrayList<>();

    for (int budget : Budgetlist) {

      INDArrayIndex[] indices = new INDArrayIndex[2 + K];
      indices[0] = NDArrayIndex.point(n-1);
      indices[1] = NDArrayIndex.point(budget);

      // Append k "all" indices:
      for (int i = 0; i < K; i++) {
        indices[i + 2] = NDArrayIndex.all();
      }

//      INDArray slice = D.get(indices);
//      System.out.println(budget + " " + slice.maxNumber().doubleValue());

      String[] results = recurTrace(symbols, D, record, K, bestR, n, budget, idxlist);

//      double[] cleandata = cleanDataCentroids(data, symbols, results,
//               centroids, n);

      double[] cleandata = cleanDataLR(data, symbols, results, breakpoints,
          centroids, K, n, countMap, posMap);

      TimeSeries resultSeires = assist.List2TS(cleandata);
      resultSeriesList.add(resultSeires);
    }

    return resultSeriesList;
  }

  public TimeSeries mainAkaneAuto() {
    Assist assist = new Assist();
    double[] data = assist.TS2List(timeseries);
    int n = data.length;
    double maxValue = Arrays.stream(data).max().getAsDouble();
    double minValue = Arrays.stream(data).min().getAsDouble();

    double minresult = Double.POSITIVE_INFINITY;
    int bestR = -1;
    for(int testr = 2; testr < 11; testr++){
      double[] centroids = clusterData(data, testr);
      Arrays.sort(centroids);
      double[] breakpoints = getBreakpoints(centroids, minValue, maxValue);
      String[] symbols = symbolizeData(data, breakpoints);
      double result = computeDaviesBouldinIndex(data, symbols, centroids);
//      System.out.println(testr + " " + result);
      if(minresult > result){
        minresult = result;
        bestR = testr;
      }
    }

    System.out.println("Choose Cluster Number R = " + bestR);
    double[] centroids = clusterData(data, bestR);
    Arrays.sort(centroids);
    double[] breakpoints = getBreakpoints(centroids, minValue, maxValue);
//    for(int i = 0; i < breakpoints.length; i++){
//      System.out.println(breakpoints[i]);
//    }
    String[] symbols = symbolizeData(data, breakpoints);
    String[] idxlist = getSymbolSet(bestR);

    Map<ArrayList<String>, Integer> countMap = getCount(symbols, K + 1);
    Map<ArrayList<String>, ArrayList<Integer>> posMap = getPos(symbols, K + 1);

    int maxBudget = (5 * n) / 10;

    int[] d_shape = new int[K + 2];
    d_shape[0] = n;
    d_shape[1] = maxBudget + 1;
    for (int i = 2; i < K + 2; i++) {
      d_shape[i] = bestR;
    }
    INDArray D = Nd4j.zeros(d_shape).subi(Double.POSITIVE_INFINITY);
    INDArray record = Nd4j.zeros(d_shape);
    ArrayList<String> chrlist = new ArrayList<>();
    recurInitial(D, K, K, symbols, bestR, countMap, n, idxlist, chrlist, maxBudget);

    ArrayList<Double> likelihoodlist = new ArrayList<>();
    likelihoodlist.add(Double.NEGATIVE_INFINITY);
    ArrayList<Double> growthratelist = new ArrayList<>();

    int expectedBudget = maxBudget;
    double prev_alpha = -1;
    int nochange_cnt = 1;
    for (int w = 0; w < maxBudget + 1; w++) {
//      if(w % 10 == 0){
//        System.out.println(w);
//      }
      for (int i = K; i < n; i++) {
        recurDP(D, record, i, K, K, symbols, bestR,
            countMap, n, idxlist, chrlist, w, -1); // any value for prevw is ok here
      }

      INDArrayIndex[] indices = new INDArrayIndex[2 + K];
      indices[0] = NDArrayIndex.point(n-1);
      indices[1] = NDArrayIndex.point(w);

      // Append k "all" indices:
      for (int i = 0; i < K; i++) {
        indices[i + 2] = NDArrayIndex.all();
      }

      INDArray slice = D.get(indices);

      likelihoodlist.add(slice.maxNumber().doubleValue());
      growthratelist.add(likelihoodlist.get(w + 1) - likelihoodlist.get(w));

      if (w >= K) {
        double omega = 0;
        double alpha = 0;
        double cur_tau = (w * 1.0) / (n * 1.0);
        double cur_omega = Math.log((5 * Math.pow(1.0 - cur_tau, K + 1) + 1) / (5 * (cur_tau / ((bestR - 1) * 1.0)) * Math.pow(1.0 - cur_tau, K) + 1));
        double cur_alpha = growthratelist.get(w);
        double cur_total = likelihoodlist.get(w + 1);

        // System.out.println(w + " " + cur_omega + " " + cur_alpha + " " + cur_total);

        for (int i = w - K; i <= w; i++) {
          double tau = (i * 1.0) / (n * 1.0);
          omega += Math.log((5 * Math.pow(1.0 - tau, K + 1) + 1) / (5 * (tau / ((bestR - 1) * 1.0)) * Math.pow(1.0 - tau, K) + 1));
          alpha += growthratelist.get(i);
        }
      //  System.out.println(w+" "+omega+" "+alpha);
       if (omega > alpha) {
         expectedBudget = w;
         System.out.println("Auto select budget " + expectedBudget);
         break;
       }
       if(prev_alpha == alpha){
         nochange_cnt += 1;
         if(nochange_cnt > K && alpha-omega < 0.5*(K+1)){
           expectedBudget = w;
           System.out.println("Auto select budget " + expectedBudget);
           break;
         }
       }else{
         nochange_cnt = 1;
       }

       prev_alpha = alpha;
      }
    }

    INDArrayIndex[] indices = new INDArrayIndex[2 + K];
    indices[0] = NDArrayIndex.point(n-1);
    indices[1] = NDArrayIndex.point(expectedBudget);

    // Append k "all" indices:
    for (int i = 0; i < K; i++) {
      indices[i + 2] = NDArrayIndex.all();
    }

    INDArray slice = D.get(indices);
//    System.out.println(slice.maxNumber().doubleValue());

    String[] results = recurTrace(symbols, D, record, K, bestR, n, expectedBudget, idxlist);

//      double[] cleandata = cleanDataCentroids(data, symbols, results,
//               centroids, n);

    double[] cleandata = cleanDataLR(data, symbols, results, breakpoints,
        centroids, K, n, countMap, posMap);

    TimeSeries resultSeires = assist.List2TS(cleandata);
    return resultSeires;
  }

}
