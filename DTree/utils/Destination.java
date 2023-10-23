package utils;

public enum Destination {
    C("Cherbourg"),Q("Queenstown"),S("Southampton");
    private String value;
    Destination(String value){
        this.value=value;
    }
    @Override
    public String toString() {
        return value;
    }
}
