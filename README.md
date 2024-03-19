# Akane

Code release of "Akane: Perplexity-Guided Time Series Data Cleaning". (Accpeted to SIGMOD 2024.)

## Settings and Dependencies

**OS:** Ubuntu 18.04 with Linux Kernel 4.15.0-55-generic

**CPU:** Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

**JDK:** Java 11 Amazon Corretto version 11.0.20

**Libraries:** commons-math3-3.6; weka-3.8; javaml-0.1; nd4j-native-1.0.0-M2

*We make sure that our algorithms can work on the given settings and dependencies, but we haven't tested them on others.*

## File Descriptions

`Akane/src/akane/` : code files of our algorithms.

- `entity/TimePoint.java` and `entity/TimeSeries.java` , which include the TimePoint class and TimeSeries class. 
- `test/AkaneTest.java` , in which you can evaluate our algorithms.
- `util/Assist.java` , which is the utility class and includes many useful tools for our algorithms.
- `Akane.java` : represents the algorithm **Akane** in the paper.
  - Symbolization: K-Means.
  - Probability Calculation: no homomorphic aggregation.
  - Likelihood Optimization: global optimization with Algorithm 1 in Section 3.4.1.
  - Reconstruction: linear-regression.
- `AkaneHeuristic.java` : represents the algorithm **AkaneH** in the paper.
  - Symbolization: K-Means.
  - Probability Calculation: no homomorphic aggregation.
  - Likelihood Optimization: greedy-based heuristic algorithm introduced in Section 4.2.
  - Reconstruction: linear-regression.
- `AkaneHeuristicHomo.java` : represents the algorithm **AkaneH+** in the paper.
  - Symbolization: uniform symbolization.
  - Probability Calculation: <u>with</u> homomorphic aggregation.
  - Likelihood Optimization: greedy-based heuristic algorithm introduced in Section 4.2.
  - Reconstruction: linear-regression.

`ExpData` : We provide three example datasets in this repo. Files are named as "NameLength-DirtyRate-Number.data", such as "ca10k-0.2-D1.data" standing for <u>D1 dataset CA whose length is 10k and dirty rate is 0.2</u>. The schema of the data file contains three columns: timestamp (Col 1): the timestamp of the data; dirty data (Col 2): the observation; truth data (Col 3): the truth.

## Datasets

**D1 dataset CA:** hourly energy consumption data inMWH (Megawatt-hours) for all four utilities in California from 2014-01-01 to 2018-12-31, from [California ISO](http://www.caiso.com/planning/Pages/ReliabilityRequirements/Default.aspx).

**D2 dataset Romania:** hourly energy consumption data in Romania from 2019-01-01 to 2023-03-12, from [Kaggle](https://www.kaggle.com/datasets/stefancomanita/hourly-electricity-consumption-and-production).

## Main Codes

**Methods:**

```java
Akane akane = new Akane(dirtySeries, K, Budgetlist);
Akane akaneauto = new Akane(dirtySeries, K);
```

```java
AkaneHeuristic akaneheuristic = new AkaneHeuristic(dirtySeries, K, BudgetlistHeuristic);
AkaneHeuristic akaneheuristicauto = new AkaneHeuristic(dirtySeries, K);
```

```java
AkaneHeuristicHomo akaneheuristichomo = new AkaneHeuristicHomo(dirtySeries, R, K, BudgetlistHeuristic, Epsilon);
AkaneHeuristicHomo akaneheuristichomoauto = new AkaneHeuristicHomo(dirtySeries, R, K, Epsilon);
```

**Inputs:**

```java
TimeSeries dirtySeries; // the dirty time series object in TimeSeries class
int K = 3; // Markov chain order
int[] Budgetlist; // same below
int[] BudgetlistHeuristic; // input int array that stores budget. For example, if this array is [10, 20], the algorithm will calculate and return two resultSeries respectively under budget 10 and 20 in the resultSeriesList. This can help use find the optimal result.
int R = 25; // uniform symbolization number r (for K-Means, we directly calculate r in the algorithm using DBI)
double Epsilon = 0.01; // homomorphic pattern aggregation distance threshold. Here the Epsilon is actually half of the value introduced in the paper. We use this value to initial homomorphic set, while in the paper we use Epsilon/2. So the predefined Epsilon = 0.02 in the paper equals to Epsilon = 0.01 here.
```

**Data Cleaning:**

```java
ArrayList<TimeSeries> resultSeriesList = akane.mainAkane();
TimeSeries resultSeries = akaneauto.mainAkaneAuto();
```

```java
ArrayList<TimeSeries> resultSeriesListHeuristic = akaneheuristic.mainAkaneHeuristic();
TimeSeries resultSeriesheuristic = akaneheuristicauto.mainAkaneHeuristicAuto();
```

```java
ArrayList<TimeSeries> resultSeriesListHeuristicHomo = akaneheuristichomo.mainAkaneHeuristicHomo();
TimeSeries resultSeriesHeuristicHomoAuto = akaneheuristichomoauto.mainAkaneHeuristicHomoAuto();
```

For `.mainName()` , it returns a list of resultSeries corresponding to different budgets in BudgetList. By sequentially scanning this resultSeriesList and calculating the RMSE between the truthSeries and the element in the List, we can get the optimal budget and RMSE.

For `.mainNameAuto()`, it is actually the algorithm using the automatic budget selection strategy introduced in Section 3.4.2 in our paper, and we compare them to the baselines in the experiments. They will return the resultSeries corresponding to the auto-selected budget.

Besides, function `cleanDataLR()` in the codes is the linear regression method used for the reconstruction. You can compare it to the function `cleanDataCentroids()` to evaluate the effectiveness of the LR reconstruction strategy, which is not included in the experiments due to the length limitation of the submitted paper. 

**Outputs:**

```java
ArrayList<TimeSeries> resultSeriesList;
ArrayList<TimeSeries> resultSeriesListHeuristic;
ArrayList<TimeSeries> resultSeriesListHeuristicHomo;
// or
TimeSeries resultSeries;
TimeSeries resultSeriesheuristic;
TimeSeries resultSeriesHeuristicHomoAuto;
```

## To Start With

After having prepared all dependencies, please first modify `public static String PATH = "./Akane/ExpData/";` in `util/Assist.java` , to make sure the code can correctly find the data for experiments, and then run `test/AkaneTest.java` to test our algorithms. This test is a simple demo, and you can modify everything to test every scenarios.