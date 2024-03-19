package akane;
import akane.util.Assist;
import akane.util.Assist.Triplet;
import java.util.Arrays;

import akane.entity.TimeSeries;

import java.util.ArrayList;
import java.util.HashSet;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.clusterers.SimpleKMeans;
import weka.core.SelectedTag;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.linear.SingularMatrixException;

public class AkaneHeuristic {

  private TimeSeries timeseries;
  private int K; // markov chain order
  private int[] Budgetlist; // different budget to clean
  private int R;

  public AkaneHeuristic(TimeSeries timeSeries, int k, int[] budgetlist){
    setTimeseries(timeSeries);
    setK(k);
    setBudgetList(budgetlist);
    setR(-1);
  }

  public AkaneHeuristic(TimeSeries timeSeries, int k, int[] budgetlist, int r){
    setTimeseries(timeSeries);
    setK(k);
    setBudgetList(budgetlist);
    setR(r);
  }

  public AkaneHeuristic(TimeSeries timeSeries, int k){
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

  public static double updateOne(int index, String chrtarget, String[] symbols,
                                 Map<ArrayList<String>, Integer> countMap, int r,
                                 int k, int n) {
    double curscore = 0.0;
    if(symbols[index].equals(chrtarget)){
      return 0;
    }

    int lb = Math.max(0, index-k);
    int rb = Math.min(index, n-k-1);

    for(int i = lb; i <= rb; i++){
      ArrayList<String> chrlist = new ArrayList<>();
      ArrayList<String> chrlistprev = new ArrayList<>();
      for(int j = 0; j <= k; j++){
        if(i+j != index){
          chrlist.add(symbols[i+j]);
        }else{
          chrlist.add(chrtarget);
        }
        chrlistprev.add(symbols[i+j]);
      }

      curscore -= getScore(chrlistprev, countMap, r, n);
      curscore += getScore(chrlist, countMap, r, n);
    }

    if(index < k){
      ArrayList<ArrayList<String>> chrheadlist = new ArrayList<>();
      ArrayList<ArrayList<String>> chrheadprevlist = new ArrayList<>();

      for(int p = k-1; p >= 0; p--){
        ArrayList<String> chrlist = new ArrayList<>();
        ArrayList<String> chrlistprev = new ArrayList<>();
        for(int i = 0; i <= p; i++){
          if(i != index){
            chrlist.add(symbols[i]);
          }else{
            chrlist.add(chrtarget);
          }
          chrlistprev.add(symbols[i]);
        }
        chrheadlist.add(chrlist);
        chrheadprevlist.add(chrlistprev);
      }

      ArrayList<ArrayList<String>> chrheadinflulist = new ArrayList<>();
      ArrayList<ArrayList<String>> chrheadinfluprevlist = new ArrayList<>();
      for(int i = 0 ; i < k-index; i++){
        chrheadinflulist.add(chrheadlist.get(i));
        chrheadinfluprevlist.add(chrheadprevlist.get(i));
      }

      for(int i = 0; i < k-index; i++){
        ArrayList<String> curr = chrheadinflulist.get(i);
        ArrayList<String> prev = chrheadinfluprevlist.get(i);

        curscore -= getScore(prev, countMap, r, n);
        curscore += getScore(curr, countMap, r, n);
      }
    }
    return curscore;
  }

  //  ArrayList<Pair<Integer, String>>
  public static ArrayList<Pair<Integer, String>> getCleanedSymbols(
      String[] symbols, Map<ArrayList<String>, Integer> countMap,
      int r, int k, int n, int budget, String[] idxlist) {
    ArrayList<Pair<Integer, String>> cleanedsymbols = new ArrayList<>();

    String[] results = Arrays.copyOf(symbols, symbols.length);
    Triplet[] maxheap = new Triplet[n];
    Assist assist = new Assist();
    int[] posRecord = new int[n];

    for(int i = 0; i < n; i++){
      double eachscore = Double.NEGATIVE_INFINITY;
      String eachchr = "-1";
      for(String chrtarget : idxlist){
        double tmp = updateOne(i, chrtarget, results, countMap, r, k ,n);
        if(tmp > eachscore){
          eachscore = tmp;
          eachchr = chrtarget;
        }
      }
      Triplet localClean = new Triplet(i, eachchr, eachscore);
//      System.out.println(bestClean.toString());
      maxheap[i] = localClean;
      posRecord[i] = i;
    }
    assist.buildHeap(maxheap, posRecord, n);

    int cnt = 0;
    while(cnt < budget){
      Triplet bestClean = maxheap[0];
      int index = bestClean.getFirst();
      String chr = bestClean.getSecond();
      if(bestClean.getThird() == 0){
        break;
      }
      results[index] = chr;
      Pair<Integer, String> pair = Pair.of(index, chr);
      cleanedsymbols.add(pair);

      int lb = Math.max(0, index-k);
      int rb = Math.min(index+k, n-1);

      for(int i = lb; i <= rb; i++){
        double eachscore = Double.NEGATIVE_INFINITY;
        String eachchr = "-1";
        for(String chrtarget : idxlist){
          double tmp = updateOne(i, chrtarget, results, countMap, r, k ,n);
          if(tmp > eachscore){
            eachscore = tmp;
            eachchr = chrtarget;
          }
        }
        Triplet localClean = new Triplet(i, eachchr, eachscore);
        assist.replaceHeap(maxheap, posRecord, i, localClean, n);
//        maxheap[posRecord[i]] = localClean;
      }
//      assist.buildHeap(maxheap, posRecord, n);

      cnt += 1;
    }

    return cleanedsymbols;
  }

  public static ArrayList<Pair<Integer, String>> getCleanedSymbolsAuto(
      String[] symbols, Map<ArrayList<String>, Integer> countMap,
      int r, int k, int n, int maxBudget, String[] idxlist) {
    ArrayList<Pair<Integer, String>> cleanedsymbols = new ArrayList<>();

    ArrayList<Double> growthratelist = new ArrayList<>();
    growthratelist.add(Double.POSITIVE_INFINITY);

    String[] results = Arrays.copyOf(symbols, symbols.length);
    Triplet[] maxheap = new Triplet[n];
    Assist assist = new Assist();
    int[] posRecord = new int[n];

    for(int i = 0; i < n; i++){
      double eachscore = Double.NEGATIVE_INFINITY;
      String eachchr = "-1";
      for(String chrtarget : idxlist){
        double tmp = updateOne(i, chrtarget, results, countMap, r, k ,n);
        if(tmp > eachscore){
          eachscore = tmp;
          eachchr = chrtarget;
        }
      }
      Triplet localClean = new Triplet(i, eachchr, eachscore);
//      System.out.println(bestClean.toString());
      maxheap[i] = localClean;
      posRecord[i] = i;
    }
    assist.buildHeap(maxheap, posRecord, n);

    int cnt = 0;
    int existingcnt = 0;
    while(cnt < maxBudget){
      Triplet bestClean = maxheap[0];
      int index = bestClean.getFirst();
      String chr = bestClean.getSecond();
      if(bestClean.getThird() == 0){
        break;
      }
      growthratelist.add(bestClean.getThird());

      results[index] = chr;
      Pair<Integer, String> pair = Pair.of(index, chr);
      cleanedsymbols.add(pair);
      cnt += 1;

      if (cnt > k) {
        double omega = 0;
        double alpha = 0;
        for (int i = cnt - k; i <= cnt; i++) {
          double tau = (i * 1.0) / (n * 1.0);
          omega += Math.log((5 * Math.pow(1.0 - tau, k + 1) + 1) / (5 * (tau/(r-1)) * Math.pow(1.0 - tau, k) + 1));
          alpha += growthratelist.get(i);
        }
//        System.out.println(cnt+" "+omega+" "+alpha);
        if (omega > alpha) {
          existingcnt += 1;
          if(existingcnt > 0.01*n){
            break;
          }
        }else{
          existingcnt = 0;
        }
      }

      int lb = Math.max(0, index-k);
      int rb = Math.min(index+k, n-1);

      for(int i = lb; i <= rb; i++){
        double eachscore = Double.NEGATIVE_INFINITY;
        String eachchr = "-1";
        for(String chrtarget : idxlist){
          double tmp = updateOne(i, chrtarget, results, countMap, r, k ,n);
          if(tmp > eachscore){
            eachscore = tmp;
            eachchr = chrtarget;
          }
        }
        Triplet localClean = new Triplet(i, eachchr, eachscore);
        assist.replaceHeap(maxheap, posRecord, i, localClean, n);
//        maxheap[posRecord[i]] = localClean;
      }
//      assist.buildHeap(maxheap, posRecord, n);

    }
    ArrayList<Pair<Integer, String>> subList = new ArrayList<>(cleanedsymbols.subList(0, cnt-existingcnt));
    System.out.println("Automatically select budget " + (cnt-existingcnt));
    return subList;
  }

  public static String[] cleanSymbols(String[] symbols, ArrayList<Pair<Integer, String>> cleanedsymbols, int budget) {
    String[] results = Arrays.copyOf(symbols, symbols.length);
    for(int i = 0; i < budget; i++){
      results[cleanedsymbols.get(i).getLeft()] = cleanedsymbols.get(i).getRight();
    }
    return results;
  }

  public static String[] cleanSymbolsAuto(String[] symbols, ArrayList<Pair<Integer, String>> cleanedsymbols) {
    String[] results = Arrays.copyOf(symbols, symbols.length);
    for (Pair<Integer, String> cleanedsymbol : cleanedsymbols) {
      results[cleanedsymbol.getLeft()] = cleanedsymbol.getRight();
    }
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

  public ArrayList<TimeSeries> mainAkaneHeuristic(){
    if(Budgetlist == null){
      System.out.println("Please provide budgetlist to clean.");
//      return null;
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

//    double maxValue = Arrays.stream(data).max().getAsDouble();
//    double minValue = Arrays.stream(data).min().getAsDouble();

    double[] breakpoints = getBreakpoints(centroids, minValue, maxValue);
    String[] symbols = symbolizeData(data, breakpoints);
    String[] idxlist = getSymbolSet(bestR);

    Map<ArrayList<String>, Integer> countMap = getCount(symbols,K+1);
    Map<ArrayList<String>, ArrayList<Integer>> posMap = getPos(symbols, K+1);

    int maxBudget = Arrays.stream(Budgetlist).max().getAsInt();
//    System.out.println(maxBudget);
    ArrayList<Pair<Integer, String>> cleanedsymbols =
        getCleanedSymbols(symbols, countMap, bestR, K, n, maxBudget, idxlist);

    ArrayList<TimeSeries> resultSeriesList = new ArrayList<>();

    int cslen = cleanedsymbols.size();

    for (int budget : Budgetlist) {

      String[] results = cleanSymbols(symbols, cleanedsymbols, Math.min(budget,cslen));

//      double[] cleandata = cleanDataCentroids(data, symbols, results,
//               centroids, n);

      double[] cleandata = cleanDataLR(data, symbols, results, breakpoints,
          centroids, K, n, countMap, posMap);

      TimeSeries resultSeires = assist.List2TS(cleandata);
      resultSeriesList.add(resultSeires);
    }

    return resultSeriesList;
  }

  public TimeSeries mainAkaneHeuristicAuto() {
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
    String[] symbols = symbolizeData(data, breakpoints);
    String[] idxlist = getSymbolSet(bestR);

    Map<ArrayList<String>, Integer> countMap = getCount(symbols,K+1);
    Map<ArrayList<String>, ArrayList<Integer>> posMap = getPos(symbols, K+1);

    int maxBudget = (3 * n) / 10;
    ArrayList<Pair<Integer, String>> cleanedsymbols =
        getCleanedSymbolsAuto(symbols, countMap, bestR, K, n, maxBudget, idxlist);

    String[] results = cleanSymbolsAuto(symbols, cleanedsymbols);

//      double[] cleandata = cleanDataCentroids(data, symbols, results,
//               centroids, n);

    double[] cleandata = cleanDataLR(data, symbols, results, breakpoints,
        centroids, K, n, countMap, posMap);

    TimeSeries resultSeries = assist.List2TS(cleandata);
    return resultSeries;
  }

}
