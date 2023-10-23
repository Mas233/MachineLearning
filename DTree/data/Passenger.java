package data;

import utils.Destination;

public class Passenger {
    private final Integer id;
    private final Boolean isSurvive;
    private final Integer seatClass;
    private final String gender;
    private final Double age;
    private final Integer sibsp;
    private final Integer parch;
    private final String ticketNumber;
    private final Double fare;
    private final Destination destination;
    public Passenger(Integer id,Boolean isSurvive,Integer seatClass,String gender,Double age,Integer sibsp,Integer parch,String ticketNumber, Double fare,Destination destination){
        this.id=id;
        this.isSurvive=isSurvive;
        this.seatClass=seatClass;
        this.gender=gender;
        this.age=age;
        this.sibsp=sibsp;
        this.parch=parch;
        this.ticketNumber=ticketNumber;
        this.fare=fare;
        this.destination=destination;
    }
    public String toString(){
        return "id:"+ id+", Survive:"+isSurvive+", seatClass:"+seatClass+", gender:"+gender+", age:"+age+", sibsp:"+sibsp+", parch:"+parch+", ticketNumber:"+ticketNumber+", fare:"+fare+", destination:"+destination.toString();
    }
}
