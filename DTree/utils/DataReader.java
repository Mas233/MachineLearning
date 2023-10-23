package utils;

import data.Passenger;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    private DataReader(){};
    private static DataReader instance = new DataReader();
    public static DataReader getInstance(){
        return instance;
    }
    public List<Passenger> readDataWithFilter(String path){
        ArrayList<Passenger> passengers = new ArrayList<>();
        try(BufferedReader br= Files.newBufferedReader(Paths.get(path))){
            String line= br.readLine();
            while((line=br.readLine())!=null){
                String[] data = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                if( data.length<12||
                        data[0].isEmpty()||
                        data[1].isEmpty()||
                        data[2].isEmpty()||
                        data[4].isEmpty()||
                        data[5].isEmpty()||
                        data[6].isEmpty()||
                        data[7].isEmpty()||
                        data[8].isEmpty()||
                        data[9].isEmpty()||
                        data[11].isEmpty()
                )
                    continue;
                Passenger passenger = new Passenger(
                        Integer.parseInt(data[0]),
                        Integer.parseInt(data[1])==1,
                        Integer.parseInt(data[2]),
                        data[4],
                        Double.parseDouble(data[5]),
                        Integer.parseInt(data[6]),
                        Integer.parseInt(data[7]),
                        data[8],
                        Double.parseDouble(data[9]),
                        Destination.valueOf(data[11])
                );
                passengers.add(passenger);
                if(passengers.size()>=1000)break;
            }

        }catch (IOException e){
            System.err.println("Error when reading data from " + path+". "+e.getMessage());
        }
        return passengers;
    }
    public static void main(String[] args) {
        DataReader.getInstance().readDataWithFilter("data/train.csv").forEach(System.out::println);
    }
}
