import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class DecisionTree{

    private double infoGain(List<Object> data, String attr){
        ArrayList<Object> attrValues = new ArrayList<>();
        HashMap<Object,Integer> attrValueCount = new HashMap<>();
        for(Object obj:data){
            if(!attrValues.contains(obj))
                attrValues.add(obj);
            attrValueCount.put(obj,attrValueCount.getOrDefault(obj,0)+1);
        }

    }
    private int[] _getValueCounts(List



























































                                  )
}
