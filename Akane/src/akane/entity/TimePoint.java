package akane.entity;

public class TimePoint {

  private long timestamp;
  private double value;
  private double modify;        // modify is in [upperbound, lowerbound]
  private boolean isModified;

  public TimePoint(long timestamp, double value) {
    setTimestamp(timestamp);
    setValue(value);
    setModify(value);
  }

  public long getTimestamp() {
    return timestamp;
  }

  public void setTimestamp(long timestamp) {
    this.timestamp = timestamp;
  }

  public double getValue() {
    return value;
  }

  public void setValue(double value) {
    this.value = value;
  }

  public double getModify() {
    return modify;
  }

  public void setModify(double modify) {
    this.modify = modify;
  }

  public boolean isModified() {
    return isModified;
  }

  public void setModified(boolean modified) {
    isModified = modified;
  }
}
