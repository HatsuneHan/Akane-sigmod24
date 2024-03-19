package akane.test;
import akane.Akane;
import akane.AkaneHeuristic;
import akane.AkaneHeuristicHomo;
import akane.util.Assist;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.sql.Time;
import java.util.Arrays;

import akane.entity.TimeSeries;
import java.util.*;
public class AkaneTest {

  public static void main(String[] args) throws FileNotFoundException {
    String expDir = "ExpParam/";
    String fileName = "ca10k-0.2-D1.data";
    String inputFileName = expDir+fileName;


    Assist assist = new Assist();
    String splitOp = ",";

    for(int tslen = 1000; tslen <= 1000; tslen += 1000){
      TimeSeries dirtySeries = assist.readData(inputFileName, 1, splitOp, tslen);
      TimeSeries truthSeries = assist.readData(inputFileName, 2, splitOp, tslen);
      System.out.println("Length: " + truthSeries.getLength());
      double rmsDirty = assist.calcRMS(truthSeries, dirtySeries);
      System.out.println("Dirty RMS error is " + rmsDirty);

      int K = 3;
      int R = 25;
      double Epsilon = 0.01;
      int[] Budgetlist= new int[truthSeries.getLength()*2/10];
      for(int i = 0; i < Budgetlist.length; i++){
        Budgetlist[i] = i;
      }

      int[] BudgetlistHeuristic = new int[truthSeries.getLength()*3/10];
      for(int i = 0; i < BudgetlistHeuristic.length; i++){
        BudgetlistHeuristic[i] = i;
      }

      long millis1;
      long millis2;
      long time;

//      Akane
      System.out.println("-------------------Akane---------------------");
      Akane akane = new Akane(dirtySeries, K, Budgetlist);
      millis1 = System.currentTimeMillis();
      ArrayList<TimeSeries> resultSeriesList = akane.mainAkane();
      int minibudget= -1;
      double minirms = Double.POSITIVE_INFINITY;
      for(int i = 0; i < Budgetlist.length; i++){
        double tmprms = assist.calcRMS(resultSeriesList.get(i), truthSeries);
        if(tmprms < minirms){
          minirms = tmprms;
          minibudget = Budgetlist[i];
        }
      }
      millis2 = System.currentTimeMillis();
      time = millis2-millis1;
      System.out.println("Mini Budget is " + minibudget);
      System.out.println("Mini Repair RMS error is " + minirms);
      System.out.println("Running Time: " + time/1000 + "s");
      System.out.println("---------------------------------------------");
      resultSeriesList = null;

//      Akane Auto
      System.out.println("------------------Akane Auto-----------------");
      Akane akaneauto = new Akane(dirtySeries, K);

      System.out.println(tslen + " Akane Auto");
      millis1 = System.currentTimeMillis();
      TimeSeries resultSeries = akaneauto.mainAkaneAuto();
      millis2 = System.currentTimeMillis();
      time = millis2-millis1;

      System.out.println("Repair RMS error is " + assist.calcRMS(resultSeries, truthSeries));
      System.out.println("Running Time: " + time/1000 + "s");
      System.out.println("---------------------------------------------");
//

//      Akane Heuristic
      System.out.println("-----------------Akane Heuristic------------------");
      AkaneHeuristic akaneheuristic = new AkaneHeuristic(dirtySeries, K, BudgetlistHeuristic);
      millis1 = System.currentTimeMillis();
      ArrayList<TimeSeries> resultSeriesListHeuristic = akaneheuristic.mainAkaneHeuristic();
      int minibudgetheuristic = -1;
      double minirmsheuristic = Double.POSITIVE_INFINITY;
      for(int i = 0; i < Budgetlist.length; i++){
        double tmprms = assist.calcRMS(resultSeriesListHeuristic.get(i), truthSeries);
        if(tmprms < minirmsheuristic){
          minirmsheuristic = tmprms;
          minibudgetheuristic = Budgetlist[i];
        }
      }
      millis2 = System.currentTimeMillis();
      time = millis2-millis1;
      System.out.println("Mini Budget is " + minibudgetheuristic);
      System.out.println("Mini Repair RMS error is " + minirmsheuristic);
      System.out.println("Running Time: " + time/1000.0 + "s");
      System.out.println("---------------------------------------------");
      resultSeriesListHeuristic = null;
//
////      Akane Heuristic Auto
      System.out.println("-----------------Akane Heuristic Auto-----------------");
      AkaneHeuristic akaneheuristicauto = new AkaneHeuristic(dirtySeries, K);

      System.out.println(tslen + " Akane Heuristic Auto");
      millis1 = System.currentTimeMillis();
      TimeSeries resultSeriesheuristic = akaneheuristicauto.mainAkaneHeuristicAuto();
      millis2 = System.currentTimeMillis();

      time = millis2-millis1;
      System.out.println("Repair RMS error is " + assist.calcRMS(resultSeriesheuristic, truthSeries));
      System.out.println("Running Time: " + time/1000.0 + "s");
      System.out.println("---------------------------------------------");
//
////      Akane Heuristic Homo
      System.out.println("-----------------Akane Heuristic Homo-----------------");
      AkaneHeuristicHomo akaneheuristichomo = new AkaneHeuristicHomo(dirtySeries, R, K, BudgetlistHeuristic, Epsilon);
      millis1 = System.currentTimeMillis();
      ArrayList<TimeSeries> resultSeriesListHeuristicHomo = akaneheuristichomo.mainAkaneHeuristicHomo();
      int minibudgetheuristichomo = -1;
      double minirmsheuristichomo = Double.POSITIVE_INFINITY;
      for(int i = 0; i < Budgetlist.length; i++){
        double tmprms = assist.calcRMS(resultSeriesListHeuristicHomo.get(i), truthSeries);
        if(tmprms < minirmsheuristichomo){
          minirmsheuristichomo = tmprms;
          minibudgetheuristichomo = Budgetlist[i];
        }
      }
      millis2 = System.currentTimeMillis();
      time = millis2-millis1;
      System.out.println("Mini Budget is " + minibudgetheuristichomo);
      System.out.println("Mini Repair RMS error is " + minirmsheuristichomo);
      System.out.println("Running Time: " + time/1000.0 + "s");
      System.out.println("---------------------------------------------");
      resultSeriesListHeuristicHomo = null;
//
////        Akane Heuristic Homo Auto
      System.out.println("-----------------Akane Heuristic Homo Auto----------------------");
      AkaneHeuristicHomo akaneheuristichomoauto = new AkaneHeuristicHomo(dirtySeries, R, K, Epsilon);

      System.out.println(tslen + " Akane Heuristic Homo Auto");
      millis1 = System.currentTimeMillis();
      TimeSeries resultSeriesHeuristicHomoAuto = akaneheuristichomoauto.mainAkaneHeuristicHomoAuto();
      millis2 = System.currentTimeMillis();

      time = millis2-millis1;
      System.out.println("Repair RMS error is " + assist.calcRMS(resultSeriesHeuristicHomoAuto, truthSeries));
      System.out.println("Running Time: " + time/1000.0 + "s");
      System.out.println("---------------------------------------------");

    }

  }
}
