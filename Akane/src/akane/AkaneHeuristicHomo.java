package akane;
import akane.util.Assist;
import akane.util.Assist.Triplet;
import java.util.Arrays;

import akane.entity.TimeSeries;
import java.util.Comparator;

import java.util.ArrayList;
import java.util.HashSet;

import java.util.HashMap;
import java.util.Map;


import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.lang3.tuple.Pair;

import net.sf.javaml.core.kdtree.KDTree;
import java.util.stream.DoubleStream;

public class AkaneHeuristicHomo {
  private TimeSeries timeseries;
  private int R; // symbol number
  private int K; // markov chain order
  private int[] Budgetlist; // different budget to clean
  private double Epsilon;

  public AkaneHeuristicHomo(TimeSeries timeSeries, int r, int k, int[] budgetlist, double epsilon){
    setTimeseries(timeSeries);
    setR(r);
    setK(k);
    setBudgetList(budgetlist);
    setEpsilon(epsilon);
  }

  public AkaneHeuristicHomo(TimeSeries timeSeries, int r, int k, double epsilon){
    setTimeseries(timeSeries);
    setR(r);
    setK(k);
    setBudgetList(null);
    setEpsilon(epsilon);
  }

  public void setTimeseries(TimeSeries timeSeries){this.timeseries = timeSeries;}
  public void setR(int r){this.R = r;}
  public void setK(int k){this.K = k;}
  public void setBudgetList(int[] budgetlist){this.Budgetlist = budgetlist;}
  public void setEpsilon(double epsilon){this.Epsilon = epsilon;}

  public Pair<Double, Double> getBaseOffset(double minValue, double maxValue, int r){
    double base = minValue;
    double offset = (maxValue-minValue+0.01)/(r*1.0); // +0.01 to make symbol number exactly = r
    return Pair.of(base, offset);
  }

  public String[] symbolizeDataEqui(double[] data, int n, double base, double offset) {
    String[] symbols = new String[n];
    for(int i = 0; i < n; i++){
      String symbol = Integer.toString((int) Math.floor((data[i]-base)/offset));
      symbols[i] = symbol;
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

  public static Map<ArrayList<String>, ArrayList<Integer>> getPos(String[] words, int k,
                                                                  Map<ArrayList<String>, ArrayList<String>> setDict) {
    Map<ArrayList<String>, ArrayList<Integer>> posMap = new HashMap<>();

    for (int i = 0; i <= words.length - k; i++) {
      ArrayList<String> pattern = new ArrayList<>();
      for (int j = 0; j < k; j++) {
        pattern.add(words[i+j]);
      }

      ArrayList<String> patternkey = setDict.get(pattern);

      ArrayList<Integer> curPos;
      if (posMap.containsKey(patternkey)) {
        curPos = posMap.get(patternkey);
      } else {
        curPos = new ArrayList<>();
      }
      curPos.add(i);
      posMap.put(patternkey, curPos);
    }

    return posMap;
  }

  public static double[] normalize(ArrayList<String> pattern){
    int len = pattern.size();
    double[] normalizedata = new double[len];
    for(int i = 0; i < len; i++){
      normalizedata[i] = Integer.parseInt(pattern.get(i));
    }
    double mean = DoubleStream.of(normalizedata).average().orElse(0.0);
    double sumOfSquares = DoubleStream.of(normalizedata)
        .map(x -> Math.pow(x - mean, 2))
        .sum();
    double std = Math.sqrt(sumOfSquares / (len-1));

    if(std == 0){
      for(int i = 0; i < len; i++){
        normalizedata[i] = 0.0;
      }
    }else{
      for(int i = 0; i < len; i++){
        normalizedata[i] = (normalizedata[i]-mean)/std;
      }
    }
    return normalizedata;
  }

  public static double getRMSDist(double[] pattern_a, double[] pattern_b){
    if (pattern_a.length != pattern_b.length) {
      throw new IllegalArgumentException("Pattern dimensions do not match");
    }

    double sum = 0.0;
    int patternlen = pattern_a.length;
    for (int i = 0; i < patternlen; i++) {
      double diff = pattern_a[i] - pattern_b[i];
      sum += diff * diff;
    }
    sum /= patternlen;

    return Math.sqrt(sum);
  }

  public void initialHomoSet(Map<ArrayList<String>, ArrayList<ArrayList<String>>> patternMap,
                             ArrayList<String> curpattern, double epsilon,
                             Map<ArrayList<String>, ArrayList<String>> setDict,
                             KDTree baseT){
    // patternMap  basepattern -> homomorphic pattern set
    // setDict     pattern -> basepattern
    if(patternMap.isEmpty()){
      ArrayList<ArrayList<String>> patternset = new ArrayList<>();
      patternset.add(curpattern);
      patternMap.put(curpattern, patternset);

      baseT.insert(normalize(curpattern), curpattern);
      setDict.put(curpattern, curpattern);
      return;
    }

    double[] curnormdata = normalize(curpattern);
    ArrayList<String> nnpattern = (ArrayList<String>) baseT.nearest(curnormdata);
    double[] nnnormdata = normalize(nnpattern);

    double dist = getRMSDist(curnormdata, nnnormdata);
    if(dist >= epsilon){
      ArrayList<ArrayList<String>> patternset = new ArrayList<>();
      patternset.add(curpattern);
      patternMap.put(curpattern, patternset);

      baseT.insert(curnormdata, curpattern);
      setDict.put(curpattern, curpattern);
    }else{
      ArrayList<ArrayList<String>> patternset = patternMap.get(nnpattern);
      patternset.add(curpattern);
      patternMap.put(nnpattern, patternset);

      setDict.put(curpattern, nnpattern);
    }

  }

  public void aggregateCount(Map<ArrayList<String>, ArrayList<ArrayList<String>>> patternMap,
                             Map<ArrayList<String>, Integer> countMap,
                             Map<ArrayList<String>, Integer> countSetMap,
                             Map<ArrayList<String>, Integer> countSetPrevMap){

    for (ArrayList<String> basepattern : patternMap.keySet()) {
      int cntSet = 0;
      int cntSetPrev = 0;

      ArrayList<ArrayList<String>> homoSet = patternMap.get(basepattern);
      for(ArrayList<String> pattern : homoSet){

        cntSet += countMap.get(pattern);

        int len = pattern.size();
        ArrayList<String> patternprev = new ArrayList<>();
        for(int i = 0; i < len-1; i++){
          patternprev.add(pattern.get((i)));
        }

        cntSetPrev += countMap.get(patternprev);
      }

      countSetMap.put(basepattern, cntSet);
      countSetPrevMap.put(basepattern, cntSetPrev);
    }
  }

  public static double getHomoScore(ArrayList<String> pattern, KDTree baseT,
                                Map<ArrayList<String>, Integer> countSetMap,
                                Map<ArrayList<String>, Integer> countSetPrevMap,
                                Map<ArrayList<String>, ArrayList<String>> setDict,
                                int r, double epsilon){

    int cnt_front;
    int cnt_all;

//    System.out.println(pattern);

    if(setDict.containsKey(pattern)){
      ArrayList<String> basepattern = setDict.get(pattern);

      cnt_all = countSetMap.get(basepattern);
      cnt_front = countSetPrevMap.get(basepattern);

    }else{
      double[] normdata = normalize(pattern);
      ArrayList<String> nnpattern = (ArrayList<String>) baseT.nearest(normdata);
      double[] nnnormdata = normalize(nnpattern);
      double dist = getRMSDist(normdata, nnnormdata);
      if(dist >= epsilon){
        cnt_all = 0;
        cnt_front = 0;
      }else{
        cnt_all = countSetMap.get(nnpattern);
        cnt_front = countSetPrevMap.get(nnpattern);
      }
    }
    double probability = (cnt_all*1.0+1.0)/(cnt_front*1.0+r*1.0);
    return Math.log(probability);
  }

  public static double updateOne(int index, String chrtarget, String[] symbols, int r,
                                 KDTree baseT, Map<ArrayList<String>, Integer> countSetMap,
                                 Map<ArrayList<String>, Integer> countSetPrevMap,
                                 Map<ArrayList<String>, ArrayList<String>> setDict,
                                 int k, int n, double epsilon) {
    double curscore = 0.0;
    if(symbols[index].equals(chrtarget)){
      return 0;
    }

    int lb = Math.max(0, index-k);
    int rb = Math.min(index, n-k-1);

    // only calculate the K+1-tuple whose start is in [lb,rb]
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

      curscore -= getHomoScore(chrlistprev, baseT, countSetMap, countSetPrevMap, setDict, r, epsilon);
      curscore += getHomoScore(chrlist, baseT, countSetMap, countSetPrevMap, setDict, r, epsilon);
    }

    return curscore;
  }

  public static ArrayList<Pair<Integer, String>> getCleanedSymbols(
      String[] symbols, KDTree baseT, Map<ArrayList<String>, Integer> countSetMap,
      Map<ArrayList<String>, Integer> countSetPrevMap,
      Map<ArrayList<String>, ArrayList<String>> setDict,
      int r, int k, int n, int budget, String[] idxlist, double epsilon) {
    ArrayList<Pair<Integer, String>> cleanedsymbols = new ArrayList<>();

    String[] results = Arrays.copyOf(symbols, symbols.length);
    Triplet[] maxheap = new Triplet[n];
    Assist assist = new Assist();
    int[] posRecord = new int[n];

    for(int i = 0; i < n; i++){
      double eachscore = Double.NEGATIVE_INFINITY;
      String eachchr = "-1";
      for(String chrtarget : idxlist){
        double tmp = updateOne(i, chrtarget, results, r, baseT, countSetMap,
            countSetPrevMap, setDict, k, n, epsilon);
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
          double tmp = updateOne(i, chrtarget, results, r, baseT, countSetMap,
              countSetPrevMap, setDict, k, n, epsilon);
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
      String[] symbols, KDTree baseT, Map<ArrayList<String>, Integer> countSetMap,
      Map<ArrayList<String>, Integer> countSetPrevMap,
      Map<ArrayList<String>, ArrayList<String>> setDict,
      int r, int k, int n, int budget, String[] idxlist, double epsilon) {
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
        double tmp = updateOne(i, chrtarget, results, r, baseT, countSetMap,
            countSetPrevMap, setDict, k, n, epsilon);
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
    while(cnt < budget){
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
          omega += Math.log((5 * Math.pow(1.0 - tau, k + 1) + 1) / (5 * (tau / ((r - 1) * 1.0)) * Math.pow(1.0 - tau, k) + 1));
          alpha += growthratelist.get(i);
        }
//        System.out.println(cnt + " " + omega + " " + alpha);
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
          double tmp = updateOne(i, chrtarget, results, r, baseT, countSetMap,
              countSetPrevMap, setDict, k, n, epsilon);
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

  public static double[] cleanDataBaseOffset(double data[], String[] symbols,
                                            String[] results, double base, double offset, int n){
    double[] cleandata = new double[n];

    for(int i = 0; i < n; i++){
      if(symbols[i].equals(results[i])){
        cleandata[i] = data[i];
      }else{
        cleandata[i] = Integer.parseInt(results[i])*offset + base + offset/2;
      }
    }

    return cleandata;
  }

  public static double getLRVal(int lb, int rb, int curi, int[] diff, int[] sumdiff,
                                Map<ArrayList<String>, Integer> countSetMap, String[] results,
                                Map<ArrayList<String>, ArrayList<Integer>> posMap, int k,
                                double defaultVal, double data[], KDTree baseT, double epsilon,
                                Map<ArrayList<String>, ArrayList<String>> setDict) {
    int max_pstart = -1;
    int max_times = 0;

    for(int i = lb; i <= rb; i++){
      ArrayList<String> pattern = new ArrayList<>();
      for (int j = i; j <= i+k; j++) {
        pattern.add(results[j]);
      }
      int times;
      if(setDict.containsKey(pattern)){
        times = countSetMap.get(setDict.get(pattern));
      }else{
        double[] normdata = normalize(pattern);
        ArrayList<String> nnpattern = (ArrayList<String>) baseT.nearest(normdata);
        double[] nnnormdata = normalize(nnpattern);
        double dist = getRMSDist(normdata, nnnormdata);
        if(dist >= epsilon){
          times = 0;
        }else{
          times = countSetMap.get(setDict.get(nnpattern));
        }
      }

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

    ArrayList<String> nnpattern = new ArrayList<>();

    if(setDict.containsKey(pattern)){
      nnpattern = setDict.get(pattern);
    }else{
      double[] normdata = normalize(pattern);
      nnpattern = (ArrayList<String>) baseT.nearest(normdata);
    }



    ArrayList<Integer> poslist = posMap.get(nnpattern);
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
                                     String[] results, double base,
                                     double offset, int k, int n, double epsilon,
                                     Map<ArrayList<String>, Integer> countSetMap, KDTree baseT,
                                     Map<ArrayList<String>, ArrayList<Integer>> posMap,
                                     Map<ArrayList<String>, ArrayList<String>> setDict) {
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

        double cleanLRVal = getLRVal(lb, rb, i, diff, sumdiff, countSetMap, results,
            posMap, k, Integer.parseInt(results[i])*offset + base + offset/2,
            data, baseT, epsilon, setDict);

        int lridx = (int) Math.floor((cleanLRVal-base)/offset);

        if(lridx < idx){
          cleandata[i] = idx * offset + base;
        }else if(lridx > idx){
          cleandata[i] = (idx+1) * offset + base;
        }else{
          cleandata[i] = cleanLRVal;
        }
      }
    }
    return cleandata;
  }

  public ArrayList<TimeSeries> mainAkaneHeuristicHomo(){
    if(Budgetlist == null){
      System.out.println("Please provide budgetlist to clean.");
      return null;
    }
    Assist assist = new Assist();
    double[] data = assist.TS2List(timeseries);
    int n = data.length;
    double maxValue = Arrays.stream(data).max().getAsDouble();
    double minValue = Arrays.stream(data).min().getAsDouble();

    String[] idxlist = getSymbolSet(R);

    Pair<Double, Double> baseoffset = getBaseOffset(minValue, maxValue, R);
    double base = baseoffset.getLeft();
    double offset = baseoffset.getRight();

    String[] symbols = symbolizeDataEqui(data, n, base, offset);

    Map<ArrayList<String>, Integer> countMap = getCount(symbols,K+1);

    Map<ArrayList<String>, ArrayList<ArrayList<String>>>
        patternMap = new HashMap<>();
    Map<ArrayList<String>, ArrayList<String>>
        setDict = new HashMap<>();
    KDTree baseT = new KDTree(K+1);

    ArrayList<Map.Entry<ArrayList<String>, Integer>> entryList = new ArrayList<>(countMap.entrySet());
    entryList.sort(Map.Entry.<ArrayList<String>, Integer>comparingByValue().reversed());

    for (Map.Entry<ArrayList<String>, Integer> entry : entryList) {
      ArrayList<String> curpattern = entry.getKey();
      if(curpattern.size() == K+1){
//        System.out.println(curpattern + " " + countMap.get(curpattern));
        initialHomoSet(patternMap, curpattern, Epsilon, setDict, baseT);
      }
    }

    Map<ArrayList<String>, ArrayList<Integer>> posMap = getPos(symbols, K+1, setDict);

    Map<ArrayList<String>, Integer> countSetMap = new HashMap<>();
    Map<ArrayList<String>, Integer> countSetPrevMap = new HashMap<>();

    aggregateCount(patternMap, countMap, countSetMap, countSetPrevMap);

    int maxBudget = Arrays.stream(Budgetlist).max().getAsInt();

    ArrayList<Pair<Integer, String>> cleanedsymbols =
        getCleanedSymbols(symbols, baseT, countSetMap, countSetPrevMap,
            setDict, R, K, n, maxBudget, idxlist, Epsilon);

    ArrayList<TimeSeries> resultSeriesList = new ArrayList<>();

    int cslen = cleanedsymbols.size();

    for (int budget : Budgetlist) {

      String[] results = cleanSymbols(symbols, cleanedsymbols, Math.min(budget,cslen));

//      double[] cleandata = cleanDataBaseOffset(data, symbols, results, base, offset, n);

      double[] cleandata = cleanDataLR(data, symbols, results, base, offset,
          K, n, Epsilon, countSetMap, baseT, posMap, setDict);

      TimeSeries resultSeires = assist.List2TS(cleandata);
      resultSeriesList.add(resultSeires);
    }

    return resultSeriesList;
  }

  public TimeSeries mainAkaneHeuristicHomoAuto(){
    Assist assist = new Assist();
    double[] data = assist.TS2List(timeseries);
    int n = data.length;
    double maxValue = Arrays.stream(data).max().getAsDouble();
    double minValue = Arrays.stream(data).min().getAsDouble();

    String[] idxlist = getSymbolSet(R);

    Pair<Double, Double> baseoffset = getBaseOffset(minValue, maxValue, R);
    double base = baseoffset.getLeft();
    double offset = baseoffset.getRight();

    String[] symbols = symbolizeDataEqui(data, n, base, offset);

    Map<ArrayList<String>, Integer> countMap = getCount(symbols,K+1);

    Map<ArrayList<String>, ArrayList<ArrayList<String>>>
        patternMap = new HashMap<>();
    Map<ArrayList<String>, ArrayList<String>>
        setDict = new HashMap<>();
    KDTree baseT = new KDTree(K+1);

    ArrayList<Map.Entry<ArrayList<String>, Integer>> entryList = new ArrayList<>(countMap.entrySet());
    entryList.sort(Map.Entry.<ArrayList<String>, Integer>comparingByValue().reversed());

    for (Map.Entry<ArrayList<String>, Integer> entry : entryList) {
      ArrayList<String> curpattern = entry.getKey();
      if(curpattern.size() == K+1){
//        ccurpattern + " " + countMap.get(curpattern));
        initialHomoSet(patternMap, curpattern, Epsilon, setDict, baseT);
      }
    }

    Map<ArrayList<String>, ArrayList<Integer>> posMap = getPos(symbols, K+1, setDict);

    Map<ArrayList<String>, Integer> countSetMap = new HashMap<>();
    Map<ArrayList<String>, Integer> countSetPrevMap = new HashMap<>();

    aggregateCount(patternMap, countMap, countSetMap, countSetPrevMap);

    int maxBudget = (3 * n) / 10;
    ArrayList<Pair<Integer, String>> cleanedsymbols =
        getCleanedSymbolsAuto(symbols, baseT, countSetMap, countSetPrevMap,
            setDict, R, K, n, maxBudget, idxlist, Epsilon);


    String[] results = cleanSymbolsAuto(symbols, cleanedsymbols);

//      double[] cleandata = cleanDataBaseOffset(data, symbols, results, base, offset, n);

    double[] cleandata = cleanDataLR(data, symbols, results, base, offset,
        K, n, Epsilon, countSetMap, baseT, posMap, setDict);

    TimeSeries resultSeries = assist.List2TS(cleandata);
    return resultSeries;
  }

}
