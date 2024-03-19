package akane.entity;

import java.util.ArrayList;

public class TimeSeries {
  private ArrayList<TimePoint> timeseries;

  public TimeSeries(ArrayList<TimePoint> timeseries) {
    setTimeseries(timeseries);
  }

  public TimeSeries() {
    setTimeseries(new ArrayList<TimePoint>());
  }

  public ArrayList<TimePoint> getTimeseries() {
    return timeseries;
  }

  public void setTimeseries(ArrayList<TimePoint> timeseries) {
    this.timeseries = timeseries;
  }

  public int getLength() {
    return timeseries.size();
  }

  public void addTimePoint(TimePoint tp) {
    timeseries.add(tp);
  }
}
